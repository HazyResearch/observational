import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import json
import os
import pickle

from emmental.scorer import Scorer
from emmental.task import EmmentalTask
from emmental.data import EmmentalDataLoader
from emmental.utils.utils import pred_to_prob, move_to_device

from end_model.dataset import (
    ObservationalDataset,
    num_classes_dict,
)
from end_model.soft_cross_entropy import SoftCrossEntropyLoss

from transforms import get_data_transforms

from tqdm import tqdm
from functools import partial


import pdb

gaze_norm_stats = {
    "cxr": (
        torch.Tensor([0.3585, 0.5312, 1.2828]),
        torch.Tensor([0.2142, 0.2206, 0.5644]),
    )
}
helper_output_dim_dict = {"loc": 9, "time": 2, "diffusivity": 2}


def ce_loss(task_name, num_classes, immediate_output_dict, Y, active):
    if num_classes > 2:
        if len(Y.shape) < 2:  # i.e. if Y not already in pred form
            Y_ = torch.Tensor(pred_to_prob(Y, num_classes))
            Y_ = move_to_device(Y_, 0)
        else:
            Y_ = Y
    else:
        Y_ = torch.stack((1 - Y, Y), axis=1)

    # active = Y != -1
    ce_loss = SoftCrossEntropyLoss().forward(
        immediate_output_dict[f"classification_module_{task_name}"][0][active],
        Y_[active],
    )

    loss = ce_loss

    return loss


def ce_loss_reweight(task_name, num_classes, immediate_output_dict, Y, active):
    if num_classes > 2:
        Y_ = torch.Tensor(pred_to_prob(Y, num_classes))
        Y_ = move_to_device(Y_, 0)
    else:
        Y_ = torch.stack((1 - Y, Y), axis=1)

    loss_outputs = SoftCrossEntropyLoss(reduction="none").forward(
        immediate_output_dict[f"classification_module_{task_name}"][0][active],
        Y_[active],
    )

    confidence_weights = torch.Tensor(
        immediate_output_dict["_input_"]["confidence"]
    ).cuda()

    return confidence_weights.dot(loss_outputs) / confidence_weights.sum()


def output(task_name, immediate_output_dict):
    return F.softmax(
        immediate_output_dict[f"classification_module_{task_name}"][0], dim=1
    )


# Two functions below for MSE gaze feature learning
def mse_loss(task_name, immediate_output_dict, Y, active):
    prediction = immediate_output_dict[f"classification_module_{task_name}"][0]
    return F.mse_loss(prediction[active], Y[active])


def mse_output(task_name, immediate_ouput_dict):
    return immediate_ouput_dict[f"classification_module_{task_name}"][0]


# this squeeze module was necessary to connect resnet encoder in emmental
class SqueezeModule(nn.Module):
    """A default identity input module that simply passes the squeezed input through."""

    def __init__(self):
        super().__init__()

    def reset_parameters(self):
        pass

    def forward(self, x):
        return x.squeeze()


# helper
def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, "w") as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def str2list(v, dim=","):
    return [t.strip() for t in v.split(dim)]


def write_to_file(path, file_name, value):
    if not isinstance(value, str):
        value = str(value)
    fout = open(os.path.join(path, file_name), "w")
    fout.write(value + "\n")
    fout.close()


def fetch_dataloaders(
    task_type,
    gaze_mtl_task,
    source,
    data_dir,
    train_scale,
    val_scale,
    seed,
    batch_size,
):

    helper_tasks = gaze_mtl_task.split("_")
    num_helper_tasks = len(helper_tasks)
    datasets = {}
    transforms = get_data_transforms(source, normalization_type="train_images")

    for split in ["train", "val", "test"]:
        datasets[split] = ObservationalDataset(
            source=source,
            task=task_type,
            gaze_mtl_task=gaze_mtl_task,
            data_dir=data_dir,
            split_type=split,
            transform=transforms[split],
            train_scale=train_scale,
            val_scale=val_scale,
            seed=seed,
        )

    task_to_label_dict = {"target": "target"}
    if task_type == "gaze_mtl":
        for i in range(num_helper_tasks):
            task_to_label_dict["helper_task_" + str(i)] = "helper_task_" + str(i)

    elif task_type == "weak_gaze":
        task_to_label_dict = {"target": "weak"}

    dataloaders = []

    for split in ["train", "val", "test"]:
        dataloaders.append(
            EmmentalDataLoader(
                task_to_label_dict=task_to_label_dict,
                dataset=datasets[split],
                split=split,
                shuffle=split == "train",
                batch_size=batch_size,
                num_workers=8,
            )
        )

    return dataloaders


def create_tasks(
    task_type,
    gaze_mtl_task,
    source,
    pretrained,
    task_weights,
    load_path=None,
):

    helper_tasks = gaze_mtl_task.split("_")
    num_helper_tasks = len(helper_tasks)
    helper_output_dims = [helper_output_dim_dict[task] for task in helper_tasks]
    num_classes = num_classes_dict[source]

    input_module = models.resnet50(pretrained=pretrained)
    num_features = 2048
    # remove last FC layer
    modules = list(input_module.children())[:-1]
    modules.append(SqueezeModule())
    cnn_module = nn.Sequential(*modules)

    if load_path is not None:
        print(f"Loading from {load_path}...")
        loaded = torch.load(load_path)
        cnn_module.load_state_dict(loaded["model"]["module_pool"]["cnn"])

    task_names = ["target"]

    weights = [1]
    num_classes_task = {"target": num_classes}
    if task_type in ["gaze_mtl", "gaze_mtl_ws"]:
        for i in range(num_helper_tasks):
            task_names.append("helper_task_" + str(i))
            num_classes_task["helper_task_" + str(i)] = helper_output_dims[i]
            weights.append(task_weights[i])

    loss_fnc = ce_loss

    tasks = [
        EmmentalTask(
            name=task_name,
            module_pool=nn.ModuleDict(
                {
                    "cnn": cnn_module,
                    f"classification_module_{task_name}": nn.Linear(
                        num_features, num_classes_task[task_name]
                    ),
                }
            ),
            task_flow=[
                {"name": "cnn", "module": "cnn", "inputs": [("_input_", "image")]},
                {
                    "name": f"classification_module_{task_name}",
                    "module": f"classification_module_{task_name}",
                    "inputs": [("cnn", 0)],
                },
            ],
            loss_func=partial(loss_fnc, task_name, num_classes_task[task_name]),
            output_func=partial(output, task_name),
            scorer=Scorer(metrics=["accuracy", "roc_auc", "precision", "recall", "f1"]),
            weight=weights[t_ndx],
        )
        for t_ndx, task_name in enumerate(task_names)
    ]

    return tasks


def save_features(save_pth, model, dataloader, task):

    model.eval()

    features_dict = {}

    encoder = model.module_pool.cnn

    for x_dict_b, y_dict_b in tqdm(dataloader, total=len(dataloader)):

        if task in ["unsup_gaze", "gaze_lstm"]:
            input_b = x_dict_b["gaze_seq"]
            img_pth_b = x_dict_b["id"]
        else:
            input_b = x_dict_b["image"]
            img_pth_b = x_dict_b["img_id"]

        feature_b = encoder(input_b.cuda()).squeeze()
        for i in range(feature_b.shape[0]):
            features_dict[img_pth_b[i]] = feature_b[i, :].detach().cpu().numpy()

    with open(os.path.join(save_pth, "train_features_dict.pkl"), "wb") as pkl_f:
        pickle.dump(features_dict, pkl_f)


def save_predictions(save_pth, emm_model, dataloaders, task):

    emm_model.eval()
    encoder = emm_model.module_pool.cnn
    target_head = emm_model.module_pool.classification_module_target

    model = nn.Sequential(encoder, target_head)

    if task == "gaze_mtl":
        helper_head = emm_model.module_pool.classification_module_helper_task_0
        helper_model = nn.Sequential(encoder, helper_head)

    for ndx, dataloader in enumerate(dataloaders):

        predictions_dict = {}

        for x_dict_b, y_dict_b in tqdm(dataloader, total=len(dataloader)):

            input_b = x_dict_b["image"]
            img_pth_b = x_dict_b["img_id"]
            true_labels_b = y_dict_b["target"]

            logits = model(input_b)
            predictions_b = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

            if task == "gaze_mtl":
                helper_true_labels_b = y_dict_b["helper_task_0"]
                logits = helper_model(input_b)
                helper_predictions_b = F.softmax(logits, dim=1).detach().cpu().numpy()

            for i in range(predictions_b.shape[0]):
                if task == "gaze_mtl":
                    predictions_dict[img_pth_b[i]] = (
                        predictions_b[i],
                        true_labels_b[i],
                        helper_predictions_b[i],
                        helper_true_labels_b[i],
                    )
                else:
                    predictions_dict[img_pth_b[i]] = (
                        predictions_b[i],
                        true_labels_b[i],
                    )

        if ndx == 0:
            file_name = "train_predictions_dict.pkl"
        elif ndx == 1:
            file_name = "val_predictions_dict.pkl"
        elif ndx == 2:
            file_name = "test_predictions_dict.pkl"

        with open(os.path.join(save_pth, file_name), "wb") as pkl_f:
            pickle.dump(predictions_dict, pkl_f)

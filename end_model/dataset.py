import os, sys
import numpy as np
import torch

from emmental.data import EmmentalDataset

from PIL import Image
import pydicom

sys.path.append("../")
from utils import (
    load_file_markers,
    load_helper_task_labels,
    load_weak_labels,
    standardize_label,
)

import pdb


num_gaze_dims_dict = {
    "none": 0,
    "loc": 9,
    "time": 1,
    "diffusivity": 1,
}
num_classes_dict = {"cxr": 2, "mets": 2, "cxr2": 2}


def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position : current_position + lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape(width, height)


class ObservationalDataset(EmmentalDataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """

    def __init__(
        self,
        source,
        task,
        gaze_mtl_task,
        data_dir,
        split_type,
        transform,
        train_scale=1,
        val_scale=0.1,
        seed=0,
    ):
        """
        Store the filenames of the seizures to use.
        Args:
            file_dir: (string) directory containing the list of file names to pull in
            split_type: (string) whether train, val, or test set
        """

        self.split_type = split_type
        self.transform = transform
        self.data_dir = data_dir
        self.source = source
        self.task = task

        # load appropriate file markers, which are a list of (image_path, label) tuples
        self.file_markers = load_file_markers(
            source,
            split_type,
            train_scale,
            val_scale,
            seed,
        )
        print(f"{len(self.file_markers)} files in {split_type} split...")

        # If Gaze-WS, load weak labels derived from gaze
        if task == "weak_gaze" and split_type == "train":
            weak_dict = load_weak_labels(source, train_scale, val_scale, seed)

        # If Gaze-MTL, load helper task labels derived from gaze
        if gaze_mtl_task:
            helper_task_labels_dict = load_helper_task_labels(source, gaze_mtl_task)

        helper_tasks = gaze_mtl_task.split("_") if gaze_mtl_task else []
        self.num_helper_tasks = len(helper_tasks)
        self.num_gaze_dims = [num_gaze_dims_dict[task] for task in helper_tasks]
        self.num_classes = num_classes_dict[source]

        X_dict = {
            "img_id": [],
            "image": [],
        }
        Y_dict = {"target": [], "weak": []}
        for i in range(self.num_helper_tasks):
            Y_dict["helper_task_" + str(i)] = []

        for idx in range(len(self.file_markers)):
            img_pth, label = self.file_markers[idx]
            img_id = img_pth.split("/")[-1]
            img_name = img_id.split(".")[0]
            if source == "cxr":
                img_id = img_pth
                img_name = img_pth
            if source == "mets":
                img_name = img_id
            X_dict["img_id"].append(img_id)

            # standardize labels so that 0 is negative and 1 is positive
            label = standardize_label(label, source)

            Y_dict["target"].append(label)
            if self.task == "gaze_mtl" and img_name in helper_task_labels_dict:
                gaze_labels = helper_task_labels_dict[img_name]
            else:
                gaze_labels = []
                for i in range(self.num_helper_tasks):
                    if self.num_gaze_dims[i] == 1:
                        gaze_labels.append(0)
                    else:
                        gaze_labels.append([0] * self.num_gaze_dims[i])

            for i in range(self.num_helper_tasks):
                Y_dict["helper_task_" + str(i)].append(gaze_labels[i])

            if split_type == "train" and task == "weak_gaze":
                if img_name in weak_dict:
                    weak_vec = weak_dict[img_name]
                else:
                    weak_vec = [0.5] * self.num_classes

                Y_dict["weak"].append(weak_vec)

        Y_dict["target"] = torch.from_numpy(np.array(Y_dict["target"]))
        Y_dict["weak"] = torch.from_numpy(np.array(Y_dict["weak"]))
        for i in range(self.num_helper_tasks):
            Y_dict["helper_task_" + str(i)] = torch.from_numpy(
                np.array(Y_dict["helper_task_" + str(i)])
            )

        super().__init__(name=source, X_dict=X_dict, Y_dict=Y_dict, uid="image")

    def __len__(self):
        return len(self.file_markers)

    def __getitem__(self, idx):

        """
        Fetch index idx image and label from dataset.
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """

        img_id = self.X_dict["img_id"][idx]
        true_label = self.Y_dict["target"][idx]
        base_dir = os.path.join(self.data_dir, self.source.upper())
        if self.source == "mets":
            # for mets there is an extra directory in the path for the image case num
            img_case_id = img_id.split("_")[1]
            img_pth = os.path.join(base_dir, "Mets_" + img_case_id)
            img_pth = os.path.join(img_pth, img_id)
        elif self.source == "cxr":
            img_pth = os.path.join(base_dir, "dicom_images", img_id)
        else:
            img_pth = os.path.join(base_dir, img_id)

        if "cxr" in self.source:
            ds = pydicom.dcmread(img_pth)
            img = ds.pixel_array
            img = Image.fromarray(np.uint8(img))
        else:
            img = Image.open(img_pth)

        img = self.transform(img)

        if img.shape[0] == 1:
            img = torch.cat([img, img, img])

        x_dict = {
            "image": img,
            "img_id": self.X_dict["img_id"][idx],
            "id": self.X_dict["img_id"][idx],
        }

        # we only want the weak label for the training set
        if self.split_type == "train" and self.task == "weak_gaze":
            weak_label = self.Y_dict["weak"][idx]
        else:
            weak_label = true_label

        # Return X and Y dictionaries depending on the task
        y_dict = {"target": true_label}
        if self.task == "weak_gaze":
            y_dict = {"target": true_label, "weak": weak_label}

        elif self.task == "gaze_mtl":
            for i in range(self.num_helper_tasks):
                y_dict["helper_task_" + str(i)] = self.Y_dict["helper_task_" + str(i)][
                    idx
                ]

        return x_dict, y_dict

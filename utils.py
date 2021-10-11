import numpy as np
import torch
import torch.nn as nn

from scipy.stats import norm
from skimage.util.shape import view_as_windows
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    recall_score,
)
import os
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from functools import partial

import pdb


def load_file_markers(
    source: str,
    split_type: str,
    train_scale: float,
    val_scale: float,
    seed: int = 0,
):
    """
    A helper function that fetches image paths and labels

    Returns: a list of file markers with (image_path,label) tuples

    Args:
        source: (str) the dataset name
        split_type: (str) whether to return [train, val, train_val, test] split
        train_scale: (float) percentage of train set to use (used for sample complexity analysis)
        val_scale: (float) the val split percentage
        seed: (int) the random seed
    """

    file_dir = os.path.join("./file_markers", source)

    if split_type in ["train", "val", "train_val"]:
        file_markers_dir = os.path.join(file_dir, "trainval_list.pkl")

        with open(file_markers_dir, "rb") as fp:
            file_markers = pickle.load(fp)

        labels = [fm[1] for fm in file_markers]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_scale, random_state=seed)

        for train_ndx, val_ndx in sss.split(np.zeros(len(labels)), labels):
            file_markers_train = [file_markers[ndx] for ndx in train_ndx]
            file_markers_val = [file_markers[ndx] for ndx in val_ndx]

        if train_scale < 1:
            # stratified shuffle split
            labels = [fm[1] for fm in file_markers_train]
            sss2 = StratifiedShuffleSplit(
                n_splits=1, test_size=train_scale, random_state=seed
            )

            for _, test_ndx in sss2.split(np.zeros(len(labels)), labels):
                file_markers_train = [file_markers_train[ndx] for ndx in test_ndx]

        if split_type == "train":
            file_markers = file_markers_train
        elif split_type == "train_val":
            file_markers = file_markers_train
            file_markers.extend(file_markers_val)
        else:
            file_markers = file_markers_val

    elif split_type == "test":
        file_markers_dir = os.path.join(file_dir, "test_list.pkl")
        with open(file_markers_dir, "rb") as fp:
            file_markers = pickle.load(fp)

    else:
        raise ValueError(f"Split type {split_type} not an option.")

    print(f"{len(file_markers)} files in {split_type} split...")

    # shuffle file markers
    np.random.seed(seed)
    np.random.shuffle(file_markers)

    return file_markers


def load_weak_labels(source: str, train_scale: float, val_scale: float, seed: int = 0):
    """
    A helper function that fetches the weak labels using the Gaze-WS method

    Returns: a dictionary of (image_id: weak_label) entries

    Args:
        source: (str) the dataset name
        train_scale: (float) percentage of train set to use (used for sample complexity analysis)
        val_scale: (float) the val split percentage
        seed: (int) the random seed
    """

    predictions, gaze_ids = run_gaze_ws(
        source,
        train_scale,
        val_scale,
        seed,
    )
    weak_dict = {gaze_id: predictions[ndx] for ndx, gaze_id in enumerate(gaze_ids)}

    return weak_dict


def load_helper_task_labels(source: str, gaze_mtl_task: str):
    """
    A helper function that fetches the helper task labels for Gaze-MTL

    Returns: a list of helper task labels

    Args:
        source: (str) the dataset name
        gaze_mtl_task: (str) the helper task (can be multiple helper tasks seperated by "_")
    """

    # pull all gaze sequences
    seqs, labels, gaze_ids = load_gaze_data(source, "train_val", 1, 0.2, 0)

    # create task_labels dict
    task_labels = {gaze_id: [] for gaze_id in gaze_ids}

    tasks = gaze_mtl_task.split("_")
    for task in tasks:

        if task == "loc":
            grid_size = 3
            heatmaps = make_heatmaps(seqs, grid_size).reshape(-1, grid_size * grid_size)
            for ndx, gaze_id in enumerate(gaze_ids):
                if labels[ndx]:
                    task_labels[gaze_id].append(np.argmax(heatmaps[ndx, :].T))
                else:
                    task_labels[gaze_id].append(0)

        elif task == "time":
            lengths = np.array([len(seq) for seq in seqs])
            mean_plus_std = np.mean(lengths) + 2 * np.std(lengths)
            lengths = np.array([min(len(seq) / mean_plus_std, 1) for seq in seqs])
            for ndx, gaze_id in enumerate(gaze_ids):
                task_labels[gaze_id].append(lengths[ndx])

        elif task == "diffusivity":
            grid_size = 10
            heatmaps = make_heatmaps(seqs, grid_size).squeeze()
            diffuse_all = apply_lf(heatmaps, partial(diffusivity, s1=2, s2=2))
            for ndx, gaze_id in enumerate(gaze_ids):
                task_labels[gaze_id].append(diffuse_all[ndx])

        else:
            raise ValueError(f"Helper task {task} not an option.")

    return task_labels


def load_gaze_data(
    source,
    split_type,
    train_scale=1,
    val_scale=0.2,
    seed=0,
    return_img_pths=False,
):
    """
    Returns: a dictionary of (gaze_id: gaze_seq) for the split type and source
    """

    gaze_dict_pth = os.path.join(
        "./gaze_data",
        source + "_gaze_data.pkl",
    )

    with open(gaze_dict_pth, "rb") as pkl_f:
        gaze_dict_all = pickle.load(pkl_f)

    # load file markers for split to know which gaze sequences to return
    if split_type == "all":
        file_markers = load_file_markers(source, "train", train_scale, val_scale, seed)
        file_markers.extend(
            load_file_markers(source, "val", train_scale, val_scale, seed)
        )
        file_markers.extend(
            load_file_markers(source, "test", train_scale, val_scale, seed)
        )
    else:
        file_markers = load_file_markers(
            source, split_type, train_scale, val_scale, seed
        )

    gaze_seqs = []
    labels = []
    gaze_ids = []
    img_pths = []
    for img_pth, lab in file_markers:
        img_pths.append(img_pth)
        labels.append(standardize_label(lab, source))
        # extract gaze_id from img_pth
        gaze_id_base = img_pth.split("/")[-1]
        gaze_id = gaze_id_base.split(".")[0]
        if source == "cxr":
            gaze_id = img_pth
        # get gaze seq
        if gaze_id in gaze_dict_all:
            if source == "mets":
                gaze_ids.append(gaze_id_base)
            else:
                gaze_ids.append(gaze_id)
            if gaze_dict_all[gaze_id] == []:
                gaze_seqs.append([[0.5, 0.5, 1]])
            else:
                gaze_seqs.append(gaze_dict_all[gaze_id])
        else:
            gaze_seqs.append([[0.5, 0.5, 1]])

    gaze_seqs = np.array(gaze_seqs)
    labels = np.array(labels)
    gaze_ids = np.array(gaze_ids)

    print(f"{len(gaze_seqs)} gaze sequences in {split_type} split...")

    if return_img_pths:
        return gaze_seqs, labels, img_pths
    return gaze_seqs, labels, gaze_ids


def standardize_label(label, source):
    # standardize labels so that 0 is negative and 1 is positive

    if source in ["cxr", "mets", "clevr", "cub", "cxr2"]:
        # label is already standardized
        standard_label = label

    elif source == "poet":
        # labels are 1-10, should be 0-9
        standard_label = label - 1

    else:
        raise ValueError(f"undefined source: {source}")

    return standard_label


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


def make_heatmaps(gaze_seqs, num_patches=8, normalize_heatmaps=False):
    all_grids = np.zeros(
        (len(gaze_seqs), 1, num_patches, num_patches), dtype=np.float32
    )
    for ndx, gaze_seq in enumerate(gaze_seqs):
        # loop through gaze seq and increment # of visits to each patch
        for (x, y, t) in gaze_seq:
            # make sure if x or y are > 1 then they are 1
            x, y = np.clip([x, y], 0.0, 0.999)
            patch_x, patch_y = int(x * num_patches), int(y * num_patches)
            all_grids[ndx, 0, patch_x, patch_y] += t
        if normalize_heatmaps:
            # Destroy total time information, as a diagnostic
            all_grids[ndx] /= np.sum(all_grids[ndx])
    return all_grids


def apply_lf(data, lf):
    # Apply a labeling function to a bunch of data
    return np.array([lf(x) for x in data])


def max_visit(heatmap, pct=0.5):
    if np.any(heatmap > np.sum(heatmap) * pct):
        return np.max(heatmap)
    return 0


def diffusivity(heatmap, s1=5, s2=5, stride=1):
    heatmap = heatmap / np.sum(heatmap)
    heatmap_windows = view_as_windows(heatmap, (s1, s2), step=stride)
    conv_results = np.tensordot(
        heatmap_windows, np.ones((s1, s2)), axes=((2, 3), (0, 1))
    )
    return np.amax(conv_results)


def unique_visits(heatmap):
    return np.sum(heatmap > 0)


def total_time(heatmap):
    return np.sum(heatmap)


def run_gaze_ws(
    source: str,
    train_scale: float,
    val_scale: float,
    seed: int = 0,
):
    """
    A helper function that runs the Gaze-WS method

    Returns: a list of the weak labels and a list of the image_ids

    Args:
        source: (str) the dataset name
        train_scale: (float) percentage of train set to use (used for sample complexity analysis)
        val_scale: (float) the val split percentage
        seed: (int) the random seed
    """

    # extract the gaze sequences of the train split
    seqs, labels, gaze_ids = load_gaze_data(
        source, "train", train_scale, val_scale, seed
    )
    # extract the gaze sequences of the val split, which will be used as "training" data in Gaze-WS
    seqs_train, y_train, _ = load_gaze_data(source, "val", train_scale, val_scale, seed)

    # Get the corresponding gaze feature hyperparameters depending on the dataset
    if source == "mets":
        grid_size = 10
        s1 = 2
        s2 = 2
        stride = 1
        view_pct = 0.3
    elif source in ["cxr", "cxr2"]:
        grid_size = 25
        s1 = grid_size
        s2 = 12
        stride = 13
        view_pct = 0

    # compute gaze features for the train split
    heatmaps = make_heatmaps(seqs, num_patches=grid_size)

    time_all = apply_lf(np.squeeze(heatmaps), total_time)
    max_visit_all = apply_lf(np.squeeze(heatmaps), partial(max_visit, pct=view_pct))
    unique_all = apply_lf(np.squeeze(heatmaps), unique_visits)
    diffusivity_all = apply_lf(
        np.squeeze(heatmaps), partial(diffusivity, s1=s1, s2=s2, stride=stride)
    )
    L = np.vstack((time_all, max_visit_all, unique_all, diffusivity_all)).T

    # compute gaze features for the val split
    heatmaps_train = make_heatmaps(seqs_train, num_patches=grid_size)

    time_train = apply_lf(np.squeeze(heatmaps_train), total_time)
    max_visit_all_train = apply_lf(
        np.squeeze(heatmaps_train), partial(max_visit, pct=view_pct)
    )
    unique_all_train = apply_lf(np.squeeze(heatmaps_train), unique_visits)
    diffusivity_all_train = apply_lf(
        np.squeeze(heatmaps_train), partial(diffusivity, s1=s1, s2=s2, stride=stride)
    )
    L_train = np.vstack(
        (time_train, max_visit_all_train, unique_all_train, diffusivity_all_train)
    ).T

    # normaliz the gaze features
    L = L - np.mean(L, 0)
    L = L / np.std(L, 0)
    L_train = L_train - np.mean(L_train, 0)
    L_train = L_train / np.std(L_train, 0)

    # run Gaze-WS method
    pred_mat_gmm_pos_test = np.zeros_like(L)
    pred_mat_gmm_pos_train = np.zeros_like(L_train)
    pred_mat_gmm_neg_test = np.zeros_like(L)
    pred_mat_gmm_neg_train = np.zeros_like(L_train)
    for feat_ind in np.arange(L.shape[1]):
        feats_train = L_train[:, feat_ind]
        feats_test = L[:, feat_ind]
        mean_pos, std_pos = norm.fit(feats_train[y_train == 1])
        mean_neg, std_neg = norm.fit(feats_train[y_train == 0])
        pred_mat_gmm_pos_test[:, feat_ind] = norm.pdf(feats_test, mean_pos, std_pos)
        pred_mat_gmm_pos_train[:, feat_ind] = norm.pdf(feats_train, mean_pos, std_pos)
        pred_mat_gmm_neg_test[:, feat_ind] = norm.pdf(feats_test, mean_neg, std_neg)
        pred_mat_gmm_neg_train[:, feat_ind] = norm.pdf(feats_train, mean_neg, std_neg)

    pred_mat_gmm_pos_test = np.prod(pred_mat_gmm_pos_test, 1) * np.mean(y_train)
    pred_mat_gmm_pos_train = np.prod(pred_mat_gmm_pos_train, 1) * np.mean(y_train)
    pred_mat_gmm_neg_test = np.prod(pred_mat_gmm_neg_test, 1) * np.mean(
        np.abs(1 - y_train)
    )
    pred_mat_gmm_neg_train = np.prod(pred_mat_gmm_neg_train, 1) * np.mean(
        np.abs(1 - y_train)
    )
    prob_preds = pred_mat_gmm_pos_train / (
        pred_mat_gmm_pos_train + pred_mat_gmm_neg_train
    )
    prob_preds = np.expand_dims(prob_preds, 0)

    best_f1 = 0
    best_thresh = 0
    for thresh in np.arange(0, 1, 0.01):
        bin_preds = (prob_preds > thresh) * 1.0
        thresh_f1 = f1_score(y_train, np.squeeze(bin_preds))
        if thresh_f1 > best_f1:
            best_f1 = thresh_f1.copy()
            best_thresh = thresh.copy()

    all_prob_preds = pred_mat_gmm_pos_test / (
        pred_mat_gmm_pos_test + pred_mat_gmm_neg_test
    )
    all_bin_preds = (all_prob_preds > best_thresh) * 1.0

    # print relevant metrics on how Gaze-WS performed
    print(
        "train size:",
        L_train.shape,
        "predict size:",
        L.shape,
        "all  roc:",
        roc_auc_score(labels, all_prob_preds),
        "all  f1:",
        f1_score(labels, all_bin_preds),
        "all  recall:",
        recall_score(labels, all_bin_preds),
        "predicted CB:",
        np.mean(all_bin_preds),
    )

    return all_bin_preds, gaze_ids


def write_to_file(path, file_name, value):
    if not isinstance(value, str):
        value = str(value)
    fout = open(os.path.join(path, file_name), "w")
    fout.write(value + "\n")
    fout.close()


def add_application_args(parser):

    parser.add_argument("--source", default="poet", type=str, help="dataset source")

    parser.add_argument(
        "--data_dir", default="original", help="Directory where image data is", type=str
    )

    parser.add_argument(
        "--train_scale", default=1, help="scale ratio for train set size", type=float
    )

    parser.add_argument(
        "--val_scale", default=0.1, help="scale ratio for train set size", type=float
    )

    parser.add_argument(
        "--task_weights",
        default=[1],
        help="weights on the aux tasks when running MTL",
        type=float,
        nargs="+",
    )

    parser.add_argument(
        "--task",
        default="original",
        help="defines the task, could be 'original', 'gaze_mtl', or 'unsup_gaze'",
        type=str,
    )

    parser.add_argument(
        "--gaze_mtl_task",
        default="none",
        help="defines the gaze mtl task, could be 'none', 'loc1', 'loc2', 'time', or 'diffusivity', and can concanenate by adding '_' between ",
        type=str,
    )

    parser.add_argument("--batch_size", type=int, default=32, help="batch size")

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Indicates if should evaluate features learnt",
    )

    parser.add_argument(
        "--transfer_learning",
        action="store_true",
        help="If true, runs tranfer learning experiment on the gaze mtl tasks",
    )

    parser.add_argument(
        "--pretrained", action="store_true", help="for pretrained model weights"
    )

    parser.add_argument("--load_cnn", type=str, default=None, help="load path")

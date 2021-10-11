from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from torchvision import transforms
import random

norm_stats = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "clevr": ([0.47097, 0.46812, 0.46185, 0], [0.08974, 0.08686, 0.09197, 1]),
    "cxr": ([0.48865], [0.24621]),
    "poet": ([0.46649, 0.4467, 0.41329], [0.26, 0.25853, 0.27346]),
}

num_channels_dict = {
    "cxr": 1,
    "cxr2": 1,
    "mets": 1,
}


def get_data_transforms(dataset_name, normalization_type="none"):
    """Get data transforms based on dataset name"""
    num_channels = num_channels_dict[dataset_name]

    if normalization_type == "none":
        mean, std = [0] * num_channels, [1] * num_channels
    elif normalization_type == "imagenet":
        if num_channels != 3:
            raise ValueError("Cannot use imagenet statistics with ≠3 channels")
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif normalization_type == "train_images":
        if dataset_name in norm_stats:
            mean, std = norm_stats[dataset_name]
        else:
            mean, std = [0] * num_channels, [1] * num_channels
    else:
        raise ValueError(f"Unknown normalization type {normalization_type}")

    eval_transform = transforms.Compose(
        [
            transforms.Resize([224, 224]),  # 384 for ViT
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_transform = eval_transform

    data_transforms = {
        "train": train_transform,
        "eval": eval_transform,
    }
    data_transforms["val"] = eval_transform
    data_transforms["test"] = eval_transform
    return data_transforms

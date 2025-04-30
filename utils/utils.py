import os
import random
import numpy as np
import torch
from torchvision import datasets, transforms


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def map_indices(dataset):
    # Map image ID → dataset index
    id_to_idx = {}
    for idx, (path, _) in enumerate(dataset.imgs):
        basename = os.path.basename(path)
        name, _ = os.path.splitext(basename)
        img_id = int(name.split('_')[-1])
        id_to_idx[img_id] = idx
    return id_to_idx


def get_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform


def transform_and_load_dataset(data_dir):
    transform = get_transform()
    # load the full ImageFolder again
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    return dataset

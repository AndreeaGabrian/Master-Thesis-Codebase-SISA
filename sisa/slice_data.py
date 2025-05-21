import json
import os
from collections import Counter

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Subset
from utils.utils import set_seed
from sklearn.model_selection import StratifiedKFold
from utils.utils import transform_and_load_dataset
import numpy as np

# Read config file
with open("utils/config.json") as f:
    cfg = json.load(f)

# Device setup
DEVICE = torch.device(
    "cuda" if (cfg["device"] == "cuda" and torch.cuda.is_available())
    else "cpu"
)

# paths & SISA params
DATA_DIR = cfg["data_dir"]
NUM_CLASSES = cfg["num_classes"]
NUM_SHARDS = cfg["num_shards"]
NUM_SLICES = cfg["num_slices"]
DATASET_NAME = cfg["dataset_name"]
SEED = 42
set_seed(SEED)
OUTPUT_DIR = cfg["output_dir"]


def get_image_ids(indices, dataset):
    ids = []
    if DATASET_NAME == "ham":
        for i in indices:
            img_path, _ = dataset.imgs[i]
            basename = os.path.basename(img_path)  # "ISIC_0027419.jpg"
            name, _ = os.path.splitext(basename)  # "ISIC_0027419"
            num_str = name.split('_')[-1]  # "0027419"
            img_id = int(num_str)  # 27419
            ids.append(img_id)

    elif DATASET_NAME in ["pathmnist", "organamnist"]:
        for i in indices:
            img_path, _ = dataset.imgs[i]
            basename = os.path.basename(img_path)  # "0027419.jpg"
            name, _ = os.path.splitext(basename)   # "0027419"
            img_id = int(name)  # 27419
            ids.append(img_id)
    else:
        raise NameError(f"Dataset name: {DATASET_NAME} is not known")
    return ids


def split_train_test_validation(dataset):
    # extract labels for stratification
    labels = [label for _, label in dataset.imgs]
    indices = list(range(len(dataset)))

    # split in train and temp (val + test)
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        indices,
        labels,
        test_size=0.3,
        stratify=labels,
        random_state=SEED
    )

    # split temp into val and test (10% + 20%)
    val_ratio = 0.1 / (0.1 + 0.2)

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=1 - val_ratio,
        stratify=temp_labels,
        random_state=SEED
    )

    train_ids = get_image_ids(train_indices, dataset)
    test_ids = get_image_ids(test_indices, dataset)
    val_ids = get_image_ids(val_indices, dataset)

    return train_ids, test_ids, val_ids, train_labels


def save_splits_to_file(train_ids, test_ids, val_ids):
    # Save split indices
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_DIR + "/train_indices.json", "w") as f:
        json.dump(train_ids, f)
    with open(OUTPUT_DIR + "/test_indices.json", "w") as f:
        json.dump(test_ids, f)
    with open(OUTPUT_DIR + "/validation_indices.json", "w") as f:
        json.dump(val_ids, f)


def save_distribution(idx_to_loc, output_path):
    rounded_idx_to_loc = {
        k: v[:3] + [round(v[3], 2)] if v[3] is not None else v
        for k, v in idx_to_loc.items()
    }
    # Save the mapping for the unlearning step
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_DIR + f"/{output_path}", "w") as f:
        json.dump(rounded_idx_to_loc, f)
    print(f"Created {NUM_SHARDS} shards each with {NUM_SLICES} slices at checkpoints/{output_path}.")


def distribute_data_slice_aware(train_indices, train_labels, unlearning_probs, output_path):
    """
    Each image is randomly assigned to a shard. Whiting each shard, samples are sorted by unlearning probability.
    Imgs with lower probability are placed into earlier slices, higher probabilities in later slices
    :param train_indices: list of img ids for training
    :param train_labels:  list of corresponding labels
    :param unlearning_probs: dictionary where (key:value) = (img_id: unlearning probability)
    :param output_path: name of the output file
    :return: None
    """
    idx_to_loc = {}

    # use StratifiedKFold to split into shards
    skf = StratifiedKFold(n_splits=NUM_SHARDS, shuffle=True, random_state=SEED)
    for k, (_, shard_idx) in enumerate(skf.split(train_indices, train_labels)):
        shard_indices = [train_indices[i] for i in shard_idx]
        # Sort shard samples by unlearning score (low to high)
        sorted_positions = sorted(range(len(shard_indices)), key=lambda i: unlearning_probs[shard_indices[i]])
        sorted_shard_indices = [shard_indices[i] for i in sorted_positions]

        slice_size = len(sorted_shard_indices) // NUM_SLICES

        for r in range(NUM_SLICES):
            start = r * slice_size
            end = (r + 1) * slice_size if r < NUM_SLICES - 1 else len(sorted_shard_indices)
            slice_indices = sorted_shard_indices[start:end]

            for idx in slice_indices:  # idx is already the int id of the image
                rel_idx = train_indices.index(idx)  # position in train_labels
                label = train_labels[rel_idx]
                unlearning_prob = unlearning_probs[idx]
                idx_to_loc[idx] = [k, r, label, unlearning_prob]

    save_distribution(idx_to_loc, output_path)


def distribute_data_shard_aware(train_indices, train_labels, unlearning_probs, output_path):
    """
    Images are sorted by unlearning likelihood, then assigned to shards based on the likelihood.
    Within each shard, samples are split randomly into slices
    :param train_indices: list of img ids for training
    :param train_labels:  list of corresponding labels
    :param unlearning_probs: dictionary where (key:value) = (img_id: unlearning probability)
    :param output_path: name of the output file
    :return: None
    """
    idx_to_loc = {}

    # Sort samples by unlearning likelihood
    sorted_positions = sorted(range(len(train_indices)), key=lambda i: unlearning_probs[train_indices[i]])
    sorted_ids = [train_indices[i] for i in sorted_positions]  # sorted dataset indices

    # Assign lowest-likelihood samples to shard 0, highest to shard K-1
    per_shard = len(sorted_ids) // NUM_SHARDS
    shard_groups = [sorted_ids[i * per_shard:(i + 1) * per_shard] for i in range(NUM_SHARDS)]

    for k, shard_indices in enumerate(shard_groups):
        np.random.shuffle(shard_indices)  # random within shard
        slice_size = len(shard_indices) // NUM_SLICES

        for r in range(NUM_SLICES):
            start = r * slice_size
            end = (r + 1) * slice_size if r < NUM_SLICES - 1 else len(shard_indices)
            slice_indices = shard_indices[start:end]

            for idx in slice_indices:
                rel_idx = train_indices.index(idx)              # get position in train_indices
                label = train_labels[rel_idx]                   # get label at that position
                unlearning_prob = unlearning_probs[idx]         # idx is dataset index
                idx_to_loc[idx] = [k, r, label, unlearning_prob]

    save_distribution(idx_to_loc, output_path)


def distribute_data_random_build_shard_slice(train_indices, train_labels, output_path):
    """
    Randomly assign each image to a shard and slice
    :param train_indices: list of img ids for training
    :param train_labels:  list of corresponding labels
    :param output_path: name of the output file
    :return: None
    :return:
    """
    # do SISA shard/slice processing only on training set
    # Data structures to hold:
    #  - shards[k] = list of Subset objects (one Subset per slice)
    #  - idx_to_loc[i] = (shard_k, slice_r) for every global index i

    idx_to_loc = {}

    # StratifiedKFold to split indices into NUM_SHARDS folds
    skf = StratifiedKFold(
        n_splits=NUM_SHARDS,
        shuffle=True,
        random_state=42
    )

    # this places each sample in a slice and a shard
    for k, (_, shard_idx) in enumerate(skf.split(train_indices, train_labels)):
        slice_size = len(shard_idx) // NUM_SLICES

        for r in range(NUM_SLICES):
            start = r * slice_size
            end = (r + 1) * slice_size if r < NUM_SLICES - 1 else len(shard_idx)
            sub_indices = shard_idx[start:end]  # these are positions in train_indices

            for i in sub_indices:
                global_idx = train_indices[i]  # actual dataset index
                label = train_labels[i]  # aligned index
                idx_to_loc[global_idx] = [k, r, label, None]

    # Save the mapping for the unlearning step
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_DIR + f"/{output_path}", "w") as f:
        json.dump(idx_to_loc, f)
    print(f"Created {NUM_SHARDS} shards each with {NUM_SLICES} slices at checkpoints/{output_path}.")


def get_unlearning_probabilities(strategy, train_indices, train_labels):
    unlearning_probs = {}
    if strategy == "random":
        unlearning_probs = {idx: float(np.random.rand()) for idx in train_indices}
    elif strategy in ["majority", "minority"]:
        # count class frequencies
        class_counts = Counter(train_labels)
        max_count = max(class_counts.values())
        # min_count = min(class_counts.values())

        if strategy == "minority":
            # higher unlearning prob for rarer classes
            unlearning_probs = {
                idx: 1.0 - (class_counts[train_labels[i]] / max_count)
                for i, idx in enumerate(train_indices)
            }
        else:
            # higher unlearning prob for more common (majority) classes
            unlearning_probs = {
                idx: (class_counts[train_labels[i]] / max(class_counts.values()))
                for i, idx in enumerate(train_indices)
            }
    return unlearning_probs


def run_split(strategy):
    # load the full ImageFolder
    dataset = transform_and_load_dataset(DATA_DIR)
    # split data intro train, test, validation
    train_ids, test_ids, val_ids, train_labels = split_train_test_validation(dataset)
    # save indices
    save_splits_to_file(train_ids, test_ids, val_ids)
    # build shards and slices and fill them with data
    output_path = f"idx_to_loc_train_k={NUM_SHARDS}_r={NUM_SLICES}.json"
    unlearning_probs = get_unlearning_probabilities("random", train_ids, train_labels)
    if strategy == "random":
        distribute_data_random_build_shard_slice(train_ids, train_labels, output_path)
    if strategy == "shard-aware":
        distribute_data_shard_aware(train_ids, train_labels, unlearning_probs, output_path)
    if strategy == "slice-aware":
        distribute_data_slice_aware(train_ids,train_labels,unlearning_probs,output_path)
run_split("random")



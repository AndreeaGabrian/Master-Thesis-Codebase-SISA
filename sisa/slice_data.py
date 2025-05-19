import json
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Subset
from utils.utils import set_seed
from sklearn.model_selection import StratifiedKFold
from utils.utils import transform_and_load_dataset

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


def distribute_data_random_build_shard_slice(dataset, train_indices, train_labels, output_path):
    # do SISA shard/slice processing only on training set
    # Data structures to hold:
    #  - shards[k] = list of Subset objects (one Subset per slice)
    #  - idx_to_loc[i] = (shard_k, slice_r) for every global index i

    # train_labels = [labels[i] for i in train_indices]
    shards = []
    idx_to_loc = {}

    # StratifiedKFold to split indices into NUM_SHARDS folds
    skf = StratifiedKFold(
        n_splits=NUM_SHARDS,
        shuffle=True,
        random_state=42
    )

    # this places each sample in a slice and a shard
    for k, (_, shard_idx) in enumerate(skf.split(train_indices, train_labels)):
        # Convert shard_idx into actual global indices
        shard_global_indices = [train_indices[i] for i in shard_idx]
        shard = Subset(dataset, shard_global_indices)

        slice_size = len(shard) // NUM_SLICES
        slices = []
        for r in range(NUM_SLICES):
            start = r * slice_size
            end = (r + 1) * slice_size if r < NUM_SLICES - 1 else len(shard)
            sub = Subset(shard, list(range(start, end)))
            slices.append(sub)

            for local_pos in range(start, end):
                global_idx = shard.indices[local_pos]
                img_path, label = dataset.imgs[global_idx]  # get class label here
                basename = os.path.basename(img_path)
                name, _ = os.path.splitext(basename)
                num_str = name.split('_')[-1]
                img_id = int(num_str)

                idx_to_loc[img_id] = [k, r, label]

        shards.append(slices)

    # Save the mapping for the unlearning step
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_DIR + f"/{output_path}", "w") as f:
        json.dump(idx_to_loc, f)
    print(f"Created {NUM_SHARDS} shards each with {NUM_SLICES} slices at checkpoints/{output_path}.")


def run_split():
    # load the full ImageFolder
    dataset = transform_and_load_dataset(DATA_DIR)
    # split data intro train, test, validation
    train_ids, test_ids, val_ids, train_labels = split_train_test_validation(dataset)
    # save indices
    save_splits_to_file(train_ids, test_ids, val_ids)
    # build shards and slices and fill them with data
    output_path = f"idx_to_loc_train_k={NUM_SHARDS}_r={NUM_SLICES}.json"
    distribute_data_random_build_shard_slice(dataset, train_ids, train_labels, output_path)



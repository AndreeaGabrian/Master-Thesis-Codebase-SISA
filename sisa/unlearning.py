import os
import json
import torch
from collections import defaultdict, Counter
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
import torch.optim as optim
from architecture.model import build_model
from evaluation.evaluate_SISA import evaluate_sisa
from utils.utils import map_indices, get_transform
from torchvision import datasets, transforms
import random
from utils.utils import get_path


config_path = get_path("utils", "config.json")
with open(config_path) as f:
    cfg = json.load(f)

OUTPUT_DIR = cfg["output_dir"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = cfg["num_classes"]
NUM_SLICES = cfg["num_slices"]
NUM_EPOCHS_PER_SLICE = cfg["num_epochs_per_slice"]
LEARNING_RATE = cfg["learning_rate"]
BATCH_SIZE = cfg["batch_size"]
training_strategy = cfg["training_strategy"]
DATASET_NAME=cfg["dataset_name"]
DISTRIBUTION = cfg["distribution"]


def random_select_image_to_unlearn(percentage, train_indices_filename):
    """
    Randomly selects a percentage of the training images to be unlearnt. E.g. 5% of the images (randomly selected)
    :param train_indices_filename: the file with the ids of training images
    :param percentage: How many images to unlearn. Float between 0 and 1
    :return: a list with the IDs of the images to unlearn
    """
    with open(get_path(train_indices_filename), 'r') as f:
        training_img_ids = json.load(f)
    n = int(len(training_img_ids) * percentage)
    unlearning_ids = random.sample(training_img_ids, n)  # randomly sample n elements without repetition
    return unlearning_ids


def unlearn(dataset, images=None, idx_to_loc_path= OUTPUT_DIR + "/idx_to_loc_train_k=5_r=3.json"):
    """
    Unlearn a specific image or a bach of images
    Parameters:
    - dataset: torchvision.datasets.ImageFolder with the full dataset loaded.
    - images: list of image IDs to unlearn
    """

    # Load current image ID - (shard, slice) map
    with open(get_path(idx_to_loc_path)) as f:
        idx_to_loc = json.load(f)

    # map image id to dataset index
    id_to_idx = map_indices(dataset)

    # determine what to unlearn
    remove_ids = set()
    affected_shards = defaultdict(set)  # a dict where the key is the affected shard and the values are the slices that contain unlearning points

    if images:  # if images is not None, thus we unlearn just one img for a bach
        for img_id in images:
            if str(img_id) in idx_to_loc:
                shard_k, slice_r, label, prob = idx_to_loc[str(img_id)]
                remove_ids.add(img_id)
                affected_shards[shard_k].add(slice_r)

    if not remove_ids:  # if the id of unlearning data is wrong
        print("No matching data found to unlearn.")
        return

    print(f"Start unlearning {len(remove_ids)} images")

    # for each affected shard, rebuild and retrain slices r to end
    for shard_k, slice_set in affected_shards.items():
        min_r = min(slice_set)  # the slice from which we have to start retraining
        print(f"\n Unlearning from shard {shard_k}, starting at slice {min_r}")

        # rebuild slices
        slices = [[] for _ in range(NUM_SLICES)]
        for img_str, (k, r, l, p) in idx_to_loc.items():
            img_id = int(img_str)
            if k == shard_k and img_id not in remove_ids:
                slices[r].append(id_to_idx[img_id])

        # create slice datasets
        slice_subsets = [Subset(dataset, idxs) for idxs in slices]

        # rebuild model
        model = build_model("resnet18", num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        # load checkpoint before min_r
        if min_r > 0:
            ckpt_path = get_path(OUTPUT_DIR + f"/{training_strategy}/shard_{shard_k}/slice_{min_r - 1}.pt")
            model.load_state_dict(torch.load(ckpt_path))
            print(f"Loaded checkpoint {ckpt_path}")
        else:
            print("No earlier checkpoint — training from scratch.")

        # retrain slices min_r to end
        for r in range(min_r, NUM_SLICES):
            loader = DataLoader(slice_subsets[r], batch_size=BATCH_SIZE, shuffle=True)
            model.train()
            for epoch in range(NUM_EPOCHS_PER_SLICE):
                for imgs, labels in loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            # save checkpoint
            out_ckpt = get_path(OUTPUT_DIR + f"/{training_strategy}/shard_{shard_k}/slice_{r}.pt")
            torch.save(model.state_dict(), out_ckpt)
            print(f"Updated checkpoint: {out_ckpt}")

    # ! update idx_to_loc by removing deleted image IDs
    for img_id in remove_ids:
        idx_to_loc.pop(str(img_id), None)

    with open(get_path(idx_to_loc_path), "w") as f:
        json.dump(idx_to_loc, f, indent=2)
    print(f"\n Unlearning ended. Updated {idx_to_loc_path}")


def progressive_unlearning_and_evaluation(dataset, train_indices_filename, eval_fn, steps=[0.05, 0.10, 0.15]):
    """
    Progressively unlearn dataset and evaluate after each step.

    :param dataset: The full torchvision dataset
    :param train_indices_filename: Path to training indices JSON file
    :param eval_fn: Evaluation function to call after unlearning
    :param steps: List of cumulative unlearning percentages
    """
    print("==== Progressive Unlearning and Evaluation ====")

    # select the full unlearning set
    with open(get_path(train_indices_filename), 'r') as f:
        all_train_ids = json.load(f)

    # group entries by class
    class_to_entries = defaultdict(list)
    for entry in all_train_ids:
        class_to_entries[entry[1]].append(entry)

    # determine total number of unlearning samples
    full_unlearn_count = int(steps[-1] * len(all_train_ids))

    # stratified sampling, proportionally sample from each class
    full_unlearn_ids = []
    label_counts = []
    for label, entries in class_to_entries.items():
        proportion = len(entries) / len(all_train_ids)
        count = int(proportion * full_unlearn_count)
        sampled = random.sample(entries, min(count, len(entries)))
        full_unlearn_ids.extend(sampled)
        label_counts.append((label, len(sampled)))  # track how many were added

    # save to a text file
    with open(f"label_counts_unlearning_{training_strategy}_{DATASET_NAME}_{DISTRIBUTION}.txt", "w") as f:
        for (label, count) in label_counts:
            f.write(f"Label/class {label}: {count}\n")

    print(f"Unlearning a total of {len(full_unlearn_ids)} images")

    previous_ids = set()
    for percent in steps:
        current_count = int(percent * len(all_train_ids))
        # current_ids = set(full_unlearn_ids[:current_count])
        current_ids = set(entry[0] for entry in full_unlearn_ids[:current_count])
        new_ids = list(current_ids - previous_ids)
        print(f"Unlearning {percent * 100:.0f}% ({len(new_ids)} images)")

        # Unlearn images
        unlearn(dataset, images=new_ids)

        # Evaluate model performance
        print(f"Evaluating after {percent * 100:.0f}% unlearning")
        eval_fn((True, percent))

        # Track which images we've already unlearned
        previous_ids = current_ids


def progressive_unlearning_and_evaluation_shard_aware(dataset, train_indices_filename, eval_fn, steps=[0.05, 0.10, 0.15]):
    """
    Progressively unlearn dataset and evaluate after each step based on unlearning probability.
    50% from prob > 0.9, 30% from >0.7, 20% from >0.5.
    """
    print("==== Progressive Unlearning and Evaluation ====")

    with open(get_path(train_indices_filename), 'r') as f:
        all_train_ids = json.load(f)  # list of [id, class, probability]

    def select_unlearning_ids_by_probability(all_train_ids, total_unlearn_count):
        # ids by prbability
        high = [e for e in all_train_ids if e[2] > 0.9]
        mid = [e for e in all_train_ids if 0.7 < e[2] <= 0.9]
        low = [e for e in all_train_ids if 0.5 < e[2] <= 0.7]

        needed_high = int(0.5 * total_unlearn_count)
        needed_mid = int(0.3 * total_unlearn_count)
        needed_low = total_unlearn_count - needed_high - needed_mid

        selected = []

        if len(high) >= needed_high:
            selected += random.sample(high, needed_high)
        else:
            selected += high
            needed_mid += needed_high - len(high)

        if len(mid) >= needed_mid:
            selected += random.sample(mid, needed_mid)
        else:
            selected += mid
            needed_low += needed_mid - len(mid)

        if len(low) >= needed_low:
            selected += random.sample(low, needed_low)
        else:
            selected += low

        return selected[:total_unlearn_count]

    # get all unlearning ids (15%)
    total_unlearn_count = int(steps[-1] * len(all_train_ids))
    full_unlearn_ids = select_unlearning_ids_by_probability(all_train_ids, total_unlearn_count)

    # tracking count per class
    class_to_entries = defaultdict(list)
    for entry in full_unlearn_ids:
        class_to_entries[entry[1]].append(entry)

    label_counts = [(label, len(entries)) for label, entries in class_to_entries.items()]
    with open(f"label_counts_unlearning_{training_strategy}_{DATASET_NAME}_{DISTRIBUTION}.txt", "w") as f:
        for (label, count) in label_counts:
            f.write(f"Label/class {label}: {count}\n")

    print(f"Unlearning a total of {len(full_unlearn_ids)} images based on probability tiers")

    # do progressive unlearning
    previous_ids = set()
    for percent in steps:
        current_count = int(percent * len(all_train_ids))
        current_ids = set(entry[0] for entry in full_unlearn_ids[:current_count])
        new_ids = list(current_ids - previous_ids)

        print(f"Unlearning {percent * 100:.0f}% ({len(new_ids)} images)")
        unlearn(dataset, images=new_ids)

        print(f"Evaluating after {percent * 100:.0f}% unlearning")
        eval_fn((True, percent))

        previous_ids = current_ids


# load full dataset
transform = get_transform()
dataset = datasets.ImageFolder(get_path("data", "HAM10000"), transform=transform)

progressive_unlearning_and_evaluation(
    dataset=dataset,
    train_indices_filename=OUTPUT_DIR + "/train_class_prob_indices.json",
    eval_fn=lambda x: evaluate_sisa(x),  # evaluation function
    steps=[0.05, 0.10, 0.15]
)

# progressive_unlearning_and_evaluation_shard_aware(
#     dataset=dataset,
#     train_indices_filename=OUTPUT_DIR + "/train_class_prob_indices.json",
#     eval_fn=lambda x: evaluate_sisa(x),  # evaluation function
#     steps=[0.05, 0.10, 0.15]
# )

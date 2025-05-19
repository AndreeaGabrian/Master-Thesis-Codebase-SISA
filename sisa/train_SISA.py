import os
import json
from collections import defaultdict
from architecture.model import build_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from utils.utils import set_seed, transform_and_load_dataset, get_transform
import time

with open("utils/config.json") as f:
    cfg = json.load(f)

# Device setup
DEVICE = torch.device(
    "cuda" if (cfg["device"] == "cuda" and torch.cuda.is_available())
    else "cpu"
)
print("Using device:", DEVICE)

# paths & SISA params
DATA_DIR = cfg["data_dir"]
OUTPUT_DIR = cfg["output_dir"]
DATASET_NAME = cfg["dataset_name"]
NUM_CLASSES = cfg["num_classes"]
NUM_SHARDS = cfg["num_shards"]
NUM_SLICES = cfg["num_slices"]

# training hyperparams
BATCH_SIZE = cfg["batch_size"]
NUM_EPOCHS_PER_SLICE = cfg["num_epochs_per_slice"]
LEARNING_RATE = cfg["learning_rate"]

# model
MODEL_NAME = cfg["model_name"]
PRETRAINED = cfg["pretrained"]

SEED = 42
set_seed(SEED)


def load_train_data():
    # Load idx_to_loc
    with open(OUTPUT_DIR + "/idx_to_loc_train.json") as f:
        idx_to_loc = json.load(f)

    dataset = transform_and_load_dataset(DATA_DIR)
    return idx_to_loc, dataset


def make_shard_slice_groups(idx_to_loc, dataset):

    # Group (shard, slice) → [indices]
    groups = defaultdict(list)

    for i, (path, _) in enumerate(dataset.imgs):
        basename = os.path.basename(path)
        img_id = int(os.path.splitext(basename)[0].split('_')[-1])
        if str(img_id) in idx_to_loc:
            k, r, c = idx_to_loc[str(img_id)]
            groups[(k, r)].append(i)

    return groups


def train_sisa(strategy="union"):
    """
    Strategy = union (default)
    This is how SISA was trained in the original paper, by unioning the slices from one shard
    Like training is done on slice 0, then 0+1, then 0+1+2 and so on

    Strategy = no-union
    Extended SISA by not unioning the slices. The weights from previous slices are used, but data is
    not repeated (unioned)
    """
    idx_to_loc, dataset = load_train_data()
    groups = make_shard_slice_groups(idx_to_loc, dataset)

    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    for k in range(NUM_SHARDS):
        print(f"\nTraining shard {k} with strategy: {strategy}")
        model = build_model(model_name="resnet18", num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for r in range(NUM_SLICES):

            if strategy == "union":
                print(f"Slice {r} union | Epochs: {NUM_EPOCHS_PER_SLICE}")
                # Union of all slices from 0 to r
                indices = []
                for r_ in range(r + 1):
                    indices.extend(groups.get((k, r_), []))

            elif strategy == "no-union":
                print(f"Slice {r} (non-union) | Epochs: {NUM_EPOCHS_PER_SLICE}")
                # Only the current slice, without union
                indices = groups.get((k, r), [])

            else:
                raise Exception("Unimplemented strategy. Options: union | no-union")

            loader = DataLoader(Subset(dataset, indices), batch_size=BATCH_SIZE, shuffle=True)
            model.train()
            for epoch in range(NUM_EPOCHS_PER_SLICE):
                total_loss = 0.0
                for i, (imgs, labels) in enumerate(loader):
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * imgs.size(0)
                    print(
                        f"[Shard {k} | Slice {r} | Epoch {epoch + 1}] Batch {i + 1}/{len(loader)} - Loss: {loss.item():.4f}")
                avg_loss = total_loss / len(loader.dataset)
                print(f"Epoch {epoch + 1}/{NUM_EPOCHS_PER_SLICE} → Loss: {avg_loss:.4f}")

            # Save checkpoint
            ckpt_dir = OUTPUT_DIR + f"/shard_{k}"
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"slice_{r}.pt"))

    # ---- stop counting time
    end_time = time.time()
    elapsed = end_time - start_time
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(f"\nTotal training time: {elapsed_str}")

    # --- Save training log
    with open(f"sisa_{strategy}_training_log_{MODEL_NAME}_{DATASET_NAME}.txt", "w") as f:
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Dataset: {DATASET_NAME}")
        f.write(f"Shards: {NUM_SHARDS}\n")
        f.write(f"Slices: {NUM_SLICES}\n")
        f.write(f"Training strategy: {strategy}\n")
        f.write(f"Epochs/slice: {NUM_EPOCHS_PER_SLICE}\n")
        f.write(f"Total time: {elapsed:.2f} seconds ({elapsed_str})\n")

    print("\n Done. All shard/slice models trained and saved")

train_sisa(strategy="union")


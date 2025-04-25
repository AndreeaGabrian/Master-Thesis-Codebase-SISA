import os
import json
import random
import numpy as np
from collections import defaultdict
from model import build_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from utils import set_seed

with open("config.json") as f:
    cfg = json.load(f)

# Device setup
DEVICE = torch.device(
    "cuda" if (cfg["device"] == "cuda" and torch.cuda.is_available())
    else "cpu"
)
print("Using device:", DEVICE)

# paths & SISA params
DATA_DIR = cfg["data_dir"]
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

# Load idx_to_loc
with open("checkpoints/idx_to_loc_train.json") as f:
    idx_to_loc = json.load(f)

# Load dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)


# Group (shard, slice) → [indices]
groups = defaultdict(list)

for i, (path, _) in enumerate(dataset.imgs):
    basename = os.path.basename(path)
    img_id = int(os.path.splitext(basename)[0].split('_')[-1])
    if str(img_id) in idx_to_loc:
        k, r = idx_to_loc[str(img_id)]
        groups[(k, r)].append(i)


criterion = nn.CrossEntropyLoss()

for k in range(NUM_SHARDS):
    print(f"\nTraining shard {k}")
    model = build_model(model_name="resnet18", num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for r in range(NUM_SLICES):
        print(f"Slice {r}")
        # Union of all slices from 0 to r
        indices = []
        for r_ in range(r+1):
            indices.extend(groups.get((k, r_), []))
        loader = DataLoader(Subset(dataset, indices), batch_size=BATCH_SIZE, shuffle=True)
        print(f"Slice {r} | Epochs: {NUM_EPOCHS_PER_SLICE}")

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
            print(f"Epoch {epoch+1}/{NUM_EPOCHS_PER_SLICE} → Loss: {avg_loss:.4f}")

        # Save checkpoint
        ckpt_dir = f"checkpoints/shard_{k}"
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"slice_{r}.pt"))

print("\n Done. All shard/slice models trained and saved")

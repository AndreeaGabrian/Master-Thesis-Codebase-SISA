import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from architecture.model import build_model
from utils.utils import set_seed, map_indices, get_transform
import time


# --- Load config
with open("utils/config.json") as f:
    cfg = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

DATA_DIR = cfg["data_dir"]
NUM_CLASSES = cfg["num_classes"]
OUTPUT_DIR = cfg["output_dir"]
DATASET_NAME = cfg["dataset_name"]
BATCH_SIZE = cfg["batch_size"]
LEARNING_RATE = cfg["learning_rate"]
MODEL_NAME = cfg["model_name"]
PRETRAINED = cfg["pretrained"]

# Match total training work from SISA
TOTAL_EPOCHS = cfg["num_slices"] * cfg["num_epochs_per_slice"]

# --- Reproducibility
SEED = 42
set_seed(SEED)

# --- Load train IDs
with open(OUTPUT_DIR + "/train_indices.json") as f:
    train_ids = json.load(f)

# --- Load dataset
transform = get_transform()
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# --- Map img_id to index
id_to_idx = map_indices(full_dataset)


# --- Build train dataset
train_indices = [id_to_idx[i] for i in train_ids]
train_dataset = Subset(full_dataset, train_indices)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Model
model = build_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=PRETRAINED).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# --- Training Loop
print(f"\n Training non-sisa monolithic model for {TOTAL_EPOCHS} epochs on {len(train_dataset)} samples")

start_time = time.time()
for epoch in range(TOTAL_EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        print(f"[Epoch {epoch+1}] Batch {i+1}/{len(train_loader)} — Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1} done — Avg loss: {avg_loss:.4f}")

# ---- stop counting time
end_time = time.time()
elapsed = end_time - start_time
elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
print(f"\nTotal training time: {elapsed_str}")

# --- Save monolithic model
os.makedirs(OUTPUT_DIR + "/monolith_non_sisa", exist_ok=True)
torch.save(model.state_dict(), OUTPUT_DIR + f"/monolith_non_sisa/final_model_{MODEL_NAME}.pt")
print(f"Monolithic non-sisa model saved to {OUTPUT_DIR}/monolith_non_sisa/final_model_{MODEL_NAME}.pt")

# --- Save training log
with open(OUTPUT_DIR + f"/monolith_non_sisa/training_log_{MODEL_NAME}.txt", "w") as f:
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Dataset: {DATASET_NAME}")
    f.write(f"Epochs: {TOTAL_EPOCHS}\n")
    f.write(f"Samples: {len(train_dataset)}\n")
    f.write(f"Total time: {elapsed:.2f} seconds ({elapsed_str})\n")
import json
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, classification_report
from utils.utils import map_indices, get_transform
from architecture.model import build_model

# --- Load config
with open("utils/config.json") as f:
    cfg = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = cfg["data_dir"]
NUM_CLASSES = cfg["num_classes"]
BATCH_SIZE = cfg["batch_size"]
# model
MODEL_NAME = cfg["model_name"]

# --- Load test IDs
with open("checkpoints/test_indices.json") as f:
    test_ids = json.load(f)

# --- Load dataset
transform = get_transform()
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# Map image ID - dataset index
id_to_idx = map_indices(dataset)

# Get test indices
test_indices = [id_to_idx[i] for i in test_ids]
test_dataset = Subset(dataset, test_indices)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Load baseline model
model = build_model(model_name=cfg["model_name"], num_classes=NUM_CLASSES, pretrained=False)
model.load_state_dict(torch.load(f"checkpoints/monolith_non_sisa/final_model_{MODEL_NAME}.pt"))
model.to(DEVICE)
model.eval()

# --- Evaluate
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels)

# Concatenate all
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)

# --- Results
acc = accuracy_score(all_labels.numpy(), all_preds.numpy())
print(f"Baseline Test Accuracy: {acc:.4f}")

# Per-class results
print("\nPer-Class Performance:\n")
report = classification_report(all_labels.numpy(), all_preds.numpy(), target_names=dataset.classes)
print(report)

# --- Save evaluation log
with open(f"evaluation_log_monolith_non_sisa_{MODEL_NAME}.txt", "w") as f:
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Test accuracy: {acc:.4f}\n")
    f.write(f"Per-Class Performance: {report}\n")

import json
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, classification_report
from utils import map_indices

from architecture.model import build_model

# --- Load config
with open("../utils/config.json") as f:
    cfg = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = cfg["data_dir"]
NUM_CLASSES = cfg["num_classes"]
BATCH_SIZE = cfg["batch_size"]

# --- Load test IDs
with open("../checkpoints/test_indices.json") as f:
    test_ids = json.load(f)

# --- Load dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# Map image ID → dataset index
id_to_idx = map_indices(dataset)


# Get test indices
test_indices = [id_to_idx[i] for i in test_ids]
test_dataset = Subset(dataset, test_indices)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Load baseline model
model = build_model(model_name=cfg["model_name"], num_classes=NUM_CLASSES, pretrained=False)
model.load_state_dict(torch.load("../checkpoints/monolith_non_sisa/final_model.pt"))
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

# Per-class report
print("\nPer-Class Performance:\n")
print(classification_report(all_labels.numpy(), all_preds.numpy(), target_names=dataset.classes))

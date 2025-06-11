import json
import torch
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from utils.utils import map_indices, get_transform
from architecture.model import build_model

# --- Load config
with open("utils/config.json") as f:
    cfg = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = cfg["data_dir"]
OUTPUT_DIR = cfg["output_dir"]
DATASET_NAME = cfg["dataset_name"]
NUM_CLASSES = cfg["num_classes"]
BATCH_SIZE = cfg["batch_size"]
# model
MODEL_NAME = cfg["model_name"]

# --- Load test IDs
with open(OUTPUT_DIR + "/test_indices.json") as f:
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
model.load_state_dict(torch.load(OUTPUT_DIR + f"/monolith_non_sisa/final_model_{MODEL_NAME}.pt"))
model.to(DEVICE)
model.eval()

# --- Evaluate
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1)  # shape [batch_size, num_classes]
        preds = torch.argmax(probs, dim=1)

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels)

# Concatenate all
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
all_probs = torch.cat(all_probs)

# --- Results
acc = accuracy_score(all_labels.numpy(), all_preds.numpy())
print(f"Baseline Test Accuracy: {acc:.4f}")

# Per-class results
print("\nPer-Class Performance:\n")
report = classification_report(all_labels.numpy(), all_preds.numpy(), target_names=dataset.classes)
print(report)


# One-vs-Rest AUC
y_true = label_binarize(all_labels.numpy(), classes=list(range(NUM_CLASSES)))
y_scores = all_probs.numpy()
auc_macro = roc_auc_score(y_true, y_scores, average="macro", multi_class="ovr")
auc_micro = roc_auc_score(y_true, y_scores, average="micro", multi_class="ovr")
class_aucs = roc_auc_score(y_true, y_scores, average=None, multi_class="ovr")

# --- Save evaluation log
with open(f"evaluation_log_monolith_non_sisa_{MODEL_NAME}_{DATASET_NAME}_CE.txt", "w") as f:
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Test accuracy: {acc:.4f}\n")
    f.write(f"Per-Class Performance: {report}\n")
    f.write(f"Macro AUC: {auc_macro}\n")
    f.write(f"Micro AUC: {auc_micro}\n")
    for i, cls in enumerate(dataset.classes):
        f.write(f"AUC for class {cls}: {class_aucs[i]:.4f}\n")

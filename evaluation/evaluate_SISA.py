import json
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from architecture.model import build_model
from sklearn.metrics import accuracy_score, classification_report
from utils import transform_and_load_dataset, map_indices

# Config
DATA_DIR = "../data/HAM10000"
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same transforms as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Full dataset
dataset = transform_and_load_dataset(DATA_DIR)

# Load test image IDs
with open("../checkpoints/test_indices.json") as f:
    test_ids = json.load(f)

# Map image ID → dataset index
id_to_idx = map_indices(dataset)

# Create test indices from IDs
test_indices = [id_to_idx[i] for i in test_ids]

# Create test dataset
test_dataset = Subset(dataset, test_indices)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Loaded test set with {len(test_dataset)} samples.")


# ------------------ LOAD MODEL SHARDS -------------
NUM_SHARDS = 5
NUM_CLASSES = 7

# Load the last slice checkpoint for each shard
models = []

for k in range(NUM_SHARDS):
    model = build_model(model_name="resnet18", num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(f"checkpoints/shard_{k}/slice_{2}.pt"))  # last slice = 2 if 3 slices total
    model.to(DEVICE)
    model.eval()
    models.append(model)

print(f"Loaded {len(models)} shard models.")


# ---------------- RUN INFERENCE AND AGGREGATE THE PREDICTIONS -------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)

        shard_outputs = []
        for model in models:
            out = model(imgs)  # logits
            prob = F.softmax(out, dim=1)  # probabilities
            shard_outputs.append(prob)

        # Average softmax outputs across shards
        avg_probs = torch.stack(shard_outputs).mean(dim=0)  # shape [batch, num_classes]

        preds = avg_probs.argmax(dim=1)  # final prediction

        all_preds.append(preds.cpu())
        all_labels.append(labels)

# Flatten everything
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)

# --------------------- COMPUTE ACCURACY ---------------------
# Overall Accuracy
acc = accuracy_score(all_labels.numpy(), all_preds.numpy())
print(f"Test Accuracy: {acc:.4f}")

# Per-class report
print("\n Per-Class Performance:\n")
print(classification_report(all_labels.numpy(), all_preds.numpy(), target_names=dataset.classes))


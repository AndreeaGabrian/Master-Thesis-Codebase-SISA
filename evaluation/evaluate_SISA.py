import json
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from architecture.model import build_model
from sklearn.metrics import accuracy_score, classification_report
from utils.utils import transform_and_load_dataset, map_indices, get_transform
from collections import defaultdict

# Config
with open("utils/config.json") as f:
    cfg = json.load(f)

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_test_data_and_models(dataset):
    # Load test image IDs
    with open("checkpoints/test_indices.json") as f:
        test_ids = json.load(f)

    # Load test image IDs
    with open("checkpoints/validation_indices.json") as f:
        validation_ids = json.load(f)

    # Map image ID → dataset index
    id_to_idx = map_indices(dataset)

    # Create test and validation indices from IDs
    test_indices = [id_to_idx[i] for i in test_ids]
    validation_indices = [id_to_idx[i] for i in validation_ids]

    # Create test dataset
    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create validation dataset
    validation_dataset = Subset(dataset, validation_indices)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Loaded test set with {len(test_dataset)} samples.")

    # Load the last slice checkpoint for each shard
    models = []
    for k in range(NUM_SHARDS):
        model = build_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False)
        model.load_state_dict(
            torch.load(f"checkpoints/shard_{k}/slice_{NUM_SLICES - 1}.pt"))  # last slice = 2 if 3 slices total
        model.to(DEVICE)
        model.eval()
        models.append(model)

    print(f"Loaded {len(models)} shard models.")
    return validation_loader, test_loader, models


# ---------------- RUN INFERENCE AND AGGREGATE THE PREDICTIONS -------------------
def get_accuracies_weights_validation(validation_loader, models):
    shard_accuracies = []
    for k, model in enumerate(models):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in validation_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        shard_accuracies.append(acc)
        # print(f"Shard {k} accuracy: {acc:.4f}")

    total_acc = sum(shard_accuracies)
    shard_weights = [acc / total_acc for acc in shard_accuracies]
    return shard_weights


def get_distribution_weights(path):
    # load idx_to_loc
    with open(f"checkpoints/{path}") as f:
        idx_to_loc = json.load(f)
    # count how many samples are in each shard and give to that shard a weight proportional to number of samples
    shard_counts = defaultdict(int)
    for k, r, _ in idx_to_loc.values():
        shard_counts[k] += 1
    total = sum(shard_counts.values())
    shard_weights = [shard_counts[k] / total for k in range(NUM_SHARDS)]
    return shard_weights


def aggregate_predictions(shard_outputs, validation_loader, models, path, strategy="soft"):
    """
    shard_outputs: list of [batch, num_classes] softmax probabilities
    strategy: the strategy to aggregate the outputs. It can be "soft", "majority", "weighted-shards", "confidence-max"
    shard_weights: optional list of weight for each shard if the aggregation strategy is "weighted-shards", length = num_shards
    """
    if strategy == "soft":  # soft vote, outputs are averaged
        avg_probs = torch.stack(shard_outputs).mean(dim=0)
        return avg_probs.argmax(dim=1)

    elif strategy == "majority":  # majority vote, the final output is the majority best probability across shards
        shard_preds = [p.argmax(dim=1) for p in shard_outputs]
        shard_preds = torch.stack(shard_preds, dim=0)
        preds, _ = torch.mode(shard_preds, dim=0)
        return preds

    elif strategy == "weighted-shards-acc":  # the shards with the higher accuracy will receive higher weights
        shard_weights = get_accuracies_weights_validation(validation_loader, models)
        weighted = sum(w * p for w, p in zip(shard_weights, shard_outputs))
        return weighted.argmax(dim=1)

    elif strategy == "weighted-shards-dist":  # the shards with more samples will receive higher weights
        shard_weights = get_distribution_weights(path)
        weighted = sum(w * p for w, p in zip(shard_weights, shard_outputs))
        return weighted.argmax(dim=1)

    elif strategy == "confidence-max":  # use only the prediction from the shards with the highes probability for that class
        confidences = [p.max(dim=1).values for p in shard_outputs]  # shape [num_shards, batch]
        best_shards = torch.stack(confidences).argmax(dim=0)        # shape [batch]
        preds = torch.stack([p.argmax(dim=1) for p in shard_outputs])  # [num_shards, batch]
        return preds.gather(0, best_shards.unsqueeze(0)).squeeze(0)

    else:
        raise NotImplementedError(f"Strategy '{strategy}' is not implemented.")


def do_inference(dataset, models, test_loader, validation_loader, path, strategy="soft"):
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

            # Aggregation step
            preds = aggregate_predictions(shard_outputs, validation_loader, models, path, strategy=strategy)

            all_preds.append(preds.cpu())
            all_labels.append(labels)

    # Flatten everything
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # --------------------- COMPUTE METRICS ---------------------
    # Overall Accuracy
    acc = accuracy_score(all_labels.numpy(), all_preds.numpy())
    print(f"Test Accuracy: {acc:.4f}")

    # Per-class report
    print("\n Per-Class Performance:\n")
    report = classification_report(all_labels.numpy(), all_preds.numpy(), target_names=dataset.classes)
    print(report)

    # --- Save evaluation log
    with open(f"evaluation_log_sisa_{MODEL_NAME}_{strategy}.txt", "w") as f:
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Num shards: {NUM_SHARDS}\n")
        f.write(f"Num slices: {NUM_SLICES}\n")
        f.write(f"Aggregation strategy: {NUM_SLICES}\n")
        f.write(f"Overall Test accuracy: {acc:.4f}\n")
        f.write(f"Per-Class Performance Metrics: {report}\n")


def evaluate_sisa():
    # Same transforms as training
    transform = get_transform()
    # Full dataset
    dataset = transform_and_load_dataset(DATA_DIR)
    # load data
    validation_loader, test_loader, models = load_test_data_and_models(dataset)
    # inference
    path = "idx_to_loc_train.json"  # I should be careful with this path
    do_inference(dataset, models, test_loader, validation_loader, path, strategy="soft")

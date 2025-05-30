import json
import numpy as np
import torch
from sklearn.preprocessing import label_binarize
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from architecture.model import build_model
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
from utils.utils import transform_and_load_dataset, map_indices, get_transform
from collections import defaultdict
from utils.utils import get_path


config_path = get_path("utils", "config.json")
with open(config_path) as f:
    cfg = json.load(f)


# paths & SISA params
DATA_DIR = cfg["data_dir"]
OUTPUT_DIR = cfg["output_dir"]
DATASET_NAME = cfg["dataset_name"]
NUM_CLASSES = cfg["num_classes"]
NUM_SHARDS = cfg["num_shards"]
NUM_SLICES = cfg["num_slices"]
training_strategy = cfg["training_strategy"]
distribution = cfg["distribution"]

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
    with open(get_path(OUTPUT_DIR, "test_indices.json")) as f:
        test_ids = json.load(f)

    # Load test image IDs
    with open(get_path(OUTPUT_DIR, "validation_indices.json")) as f:
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
    models = {}
    for k in range(NUM_SHARDS):
        model = build_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False)
        model.load_state_dict(
            torch.load(get_path(OUTPUT_DIR + f"/{training_strategy}/shard_{k}/slice_{NUM_SLICES - 1}.pt"))) # last slice = 2 if 3 slices total
        model.to(DEVICE)
        model.eval()
        models[k] = model

    print(f"Loaded {len(models)} shard models.")
    return validation_loader, test_loader, models


# ---------------- RUN INFERENCE AND AGGREGATE THE PREDICTIONS -------------------
def get_accuracies_weights_validation(validation_loader, models):
    shard_accuracies = {}
    for k, model in models.items():
        model.to(DEVICE)
        model.eval()
        all_preds = []
        all_labels = []
        for imgs, labels in validation_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)  # shape [batch_size, num_classes]
            preds = torch.argmax(probs, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        acc = accuracy_score(all_labels.numpy(), all_preds.numpy())
        shard_accuracies[k] = acc
        print(f"Shard {k} accuracy: {acc:.4f}")
    total_acc = sum(shard_accuracies.values())
    shard_weights = {k: acc / total_acc for k, acc in shard_accuracies.items()}
    return shard_weights


def get_distribution_weights(path):
    # load idx_to_loc
    with open(get_path(OUTPUT_DIR + f"/{path}")) as f:
        idx_to_loc = json.load(f)
    # count how many samples are in each shard and give to that shard a weight proportional to number of samples
    shard_counts = defaultdict(int)
    for k, r, _, prob in idx_to_loc.values():
        shard_counts[k] += 1
    total = sum(shard_counts.values())
    shard_weights = [shard_counts[k] / total for k in range(NUM_SHARDS)]
    return shard_weights


def get_or_use_weights(computed_weights, compute_fn):
    if computed_weights is None:
        return compute_fn()
    return computed_weights


def get_f1_weights_validation(
    validation_loader,
    models,
    mode="macro",             # "macro" or "per-class"
    threshold=0.1,            # used in per-class mode
    temperature=1.0,          # used in per-class mode
    epsilon=1e-4              # for smoothing
):
    """
    Returns:
    If mode == "macro": {shard_k: weight}
    If mode == "per-class": {shard_k: [weight_c0, weight_c1, ..., weight_cN]}
    """
    shard_keys = list(models.keys())
    shard_f1s = {}
    num_classes = None

    # compute predictions and F1s per shard
    for k, model in models.items():
        model.to(DEVICE)
        model.eval()
        all_preds = []
        all_labels = []
        for imgs, labels in validation_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        if mode == "macro":
            f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average="macro", zero_division=0)
            shard_f1s[k] = max(f1, epsilon)
            # print(f"Shard {k} macro F1: {f1:.4f}")
        elif mode == "per-class":
            f1_per_class = f1_score(all_labels.numpy(), all_preds.numpy(), average=None, zero_division=0)
            shard_f1s[k] = f1_per_class.tolist()
            num_classes = len(f1_per_class)
            # print(f"Shard {k} per-class F1: {f1_per_class}")
        else:
            raise ValueError(f"Unsupported mode '{mode}'. Use 'macro' or 'per-class'.")

    # If macro mode, normalize scalar F1s across shards
    if mode == "macro":
        total_f1 = sum(shard_f1s.values())
        return {k: f1 / total_f1 for k, f1 in shard_f1s.items()}

    # If per-class mode, build weight matrix
    weight_matrix = {k: [0.0] * num_classes for k in shard_keys}
    for c in range(num_classes):
        f1s = np.array([shard_f1s[k][c] for k in shard_keys])
        valid_mask = f1s >= threshold

        if not valid_mask.any():
            for k in shard_keys:
                weight_matrix[k][c] = 1.0 / len(shard_keys)
        else:
            valid_f1s = np.maximum(f1s, epsilon)
            scaled = valid_f1s / temperature
            exps = np.exp(scaled - np.max(scaled))
            softmax_weights = exps / exps.sum()

            for i, k in enumerate(shard_keys):
                weight_matrix[k][c] = softmax_weights[i] if valid_mask[i] else 0.0

    return weight_matrix


def aggregate_predictions(shard_outputs, validation_loader, models, path, computed_weights, strategy="soft"):
    """
    shard_outputs: list of [batch, num_classes] softmax probabilities
    strategy: the strategy to aggregate the outputs. It can be "soft", "majority", "weighted-shards", "confidence-max"
    shard_weights: optional list of weight for each shard if the aggregation strategy is "weighted-shards", length = num_shards
    :returns preds (logits), avg_probs(probabilities)
    """
    if strategy == "soft":  # soft (mean) vote, outputs are averaged
        avg_probs = torch.stack(shard_outputs).mean(dim=0)
        preds = avg_probs.argmax(dim=1)
        return preds, avg_probs, None

    elif strategy == "majority":  # majority vote, the final output is the majority best probability across shards
        shard_preds = [p.argmax(dim=1) for p in shard_outputs]
        shard_preds = torch.stack(shard_preds, dim=0)
        preds, _ = torch.mode(shard_preds, dim=0)
        return preds, None, None

    elif strategy in ["weighted-shards-acc", "weighted-shards-dist", "weighted-shards-f1-macro", "weighted-shards-f1-class"]:
        if strategy == "weighted-shards-acc":
            shard_weights = get_or_use_weights(computed_weights, lambda: get_accuracies_weights_validation(validation_loader, models))
        elif strategy == "weighted-shards-dist":
            shard_weights = get_or_use_weights(computed_weights, lambda: get_distribution_weights(path))
        elif strategy == "weighted-shards-f1-macro":
            shard_weights = get_or_use_weights(
                computed_weights,
                lambda: get_f1_weights_validation(validation_loader, models, mode="macro")
            )
        elif strategy == "weighted-shards-f1-class":
            weight_matrix = get_or_use_weights(
                computed_weights,
                lambda: get_f1_weights_validation(validation_loader, models, mode="per-class")
            )

        if strategy == "weighted-shards-f1-class":
            # Class-wise weighting
            all_preds = []
            for i in range(shard_outputs[0].shape[0]):  # for each sample
                classwise_sum = 0
                for k, output in enumerate(shard_outputs):
                    class_weights = torch.tensor(weight_matrix[k]).to(output.device)
                    classwise_sum += class_weights * output[i]
                all_preds.append(classwise_sum.unsqueeze(0))
            weighted = torch.cat(all_preds, dim=0)
        else:
            weighted = sum(w * p for w, p in zip(shard_weights, shard_outputs))

        preds = weighted.argmax(dim=1)

        if strategy == "weighted-shards-f1-class":
            return preds, weighted, weight_matrix
        else:
            return preds, weighted, shard_weights

    elif strategy == "confidence-max":  # use only the prediction from the shards with the highes probability for that class
        confidences = [p.max(dim=1).values for p in shard_outputs]  # shape [num_shards, batch]
        best_shards = torch.stack(confidences).argmax(dim=0)        # shape [batch]
        preds = torch.stack([p.argmax(dim=1) for p in shard_outputs])  # [num_shards, batch]
        final_preds = preds.gather(0, best_shards.unsqueeze(0)).squeeze(0)
        return final_preds, None, None

    elif strategy == "median":  # the median of the outputs is considered
        med_probs, _ = torch.stack(shard_outputs).median(dim=0)
        preds = med_probs.argmax(dim=1)
        return preds, med_probs, None

    else:
        raise NotImplementedError(f"Strategy '{strategy}' is not implemented.")


def do_inference(dataset, models, test_loader, validation_loader, path, unlearning, strategy="soft"):
    all_preds = []
    all_labels = []
    all_probs = []
    computed_weights_temp = None

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)

            shard_outputs = []
            for k, model in models.items():
                out = model(imgs)  # logits
                prob = F.softmax(out, dim=1)  # probabilities
                shard_outputs.append(prob)

            # Aggregation step
            preds, probs, computed_weights = aggregate_predictions(shard_outputs, validation_loader, models, path, computed_weights_temp, strategy=strategy)
            computed_weights_temp = computed_weights

            all_preds.append(preds.cpu())
            all_labels.append(labels)
            if probs is not None:
                all_probs.append(probs)

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

    if all_probs != []:
        all_probs = torch.cat(all_probs)
        try:
            # One-vs-Rest AUC
            y_true = label_binarize(all_labels.numpy(), classes=list(range(NUM_CLASSES)))
            y_scores = all_probs.cpu().numpy()

            auc_macro = roc_auc_score(y_true, y_scores, average="macro", multi_class="ovr")
            auc_micro = roc_auc_score(y_true, y_scores, average="micro", multi_class="ovr")
            class_aucs = roc_auc_score(y_true, y_scores, average=None, multi_class="ovr")

        except ValueError as e:
            print(f"AUC computation failed: {e}")
    else:
        auc_macro = None
        auc_micro = None
        class_aucs = None

    if unlearning[0]:
        output_filename_txt = f"evaluation_log_sisa_{DATASET_NAME}_{MODEL_NAME}_data_{distribution}_{training_strategy}_{strategy}_unlearning_{unlearning[1]}.txt"
    else:
        output_filename_txt = f"evaluation_log_sisa_{DATASET_NAME}_{MODEL_NAME}_data_{distribution}_{training_strategy}_{strategy}.txt"
    # --- Save evaluation log
    with open(get_path(output_filename_txt), "w") as f:
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Dataset name: {DATASET_NAME}\n")
        f.write(f"Num shards: {NUM_SHARDS}\n")
        f.write(f"Num slices: {NUM_SLICES}\n")
        f.write(f"Aggregation strategy: {NUM_SLICES}\n")
        f.write(f"Data distribution: {distribution}\n")
        f.write(f"-------------------------------------\n")
        f.write(f"Overall Test accuracy: {acc:.4f}\n")
        f.write(f"Per-Class Performance Metrics: {report}\n")
        f.write(f"-------------------------------------\n")
        f.write(f"Macro AUC: {auc_macro}\n")
        f.write(f"Micro AUC: {auc_micro}\n")
        for i, cls in enumerate(dataset.classes):
            if auc_macro != None:
                f.write(f"AUC for class {cls}: {class_aucs[i]:.4f}\n")
            else:
                f.write(f"AUC for class {cls}: None\n")
        if unlearning[0]:
            f.write(f"Unlearning: {unlearning[0]}, step {unlearning[1]}\n")


def evaluate_sisa(unlearning: (bool, float)):
    # Same transforms as training
    transform = get_transform()
    # Full dataset
    dataset = transform_and_load_dataset(get_path(DATA_DIR))
    # load data
    validation_loader, test_loader, models = load_test_data_and_models(dataset)
    # inference
    path = "idx_to_loc_train_k=5_r=3.json"  # I should be careful with this path
    s = ["soft", "majority", "median", "weighted-shards-dist","confidence-max","weighted-shards-acc", "weighted-shards-f1-macro", "weighted-shards-f1-class"]
    # s = ["soft"]
    for agg_strategy in s:
        print(f"Inference for agg strategy: {agg_strategy}")
        do_inference(dataset, models, test_loader, validation_loader, path, unlearning, strategy=agg_strategy)

# evaluate_sisa((False, 0.05))
import json
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from torch.utils.data import Subset
from torchvision import datasets, transforms
from utils.utils import map_indices, get_transform

# Load dataset again (if needed)
transform = get_transform()

DATA_DIR = "../data/path_mnist"
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# dataset.imgs is a list of (filepath, class_idx)
labels = [label for _, label in dataset.imgs]

# count samples per class
class_counts = Counter(labels)

# map class indices back to names
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
counts = [class_counts[i] for i in range(len(idx_to_class))]

print("Samples per class:")
for name, count in zip(class_names, counts):
    print(f"{name}: {count}")

full_class_names = {
    'akiec': "Actinic keratoses",
    'bcc': "Basal cell carcinoma",
    'bkl': "Benign keratosis-like",
    'df': "Dermatofibroma",
    'mel': "Melanoma",
    'nv': "Melanocytic nevi",
    'vasc': "Vascular lesions"
}
organmnist_classes = {
    '0': "Adrenal gland",
    '1': "Bladder",
    '2': "Brain",
    '3': "Breast",
    '4': "Esophagus",
    '5': "Eye",
    '6': "Kidney",
    '7': "Liver",
    '8': "Lung",
    '9': "Ovary",
    '10': "Pancreas"
}
pathmnist_labels = {
    '0': "Adipose",
    '1': "Background",
    '2': "Debris",
    '3': "Lymphocytes",
    '4': "Mucus",
    '5': "Smooth muscle",
    '6': "Normal colon mucosa",
    '7': "Cancer-associated stroma",
    '8': "Colorectal adenocarcinoma epithelium"
}

full_names = [pathmnist_labels[c] for c in class_names]

plt.figure(figsize=(10,6))
bars = plt.bar(full_names, counts, color="skyblue")
# Add value labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 15, int(yval), ha='center', va='bottom', fontsize=8)

plt.xlabel("Class")
plt.ylabel("Number of samples")
plt.title("Samples per class in pathMNIST dataset")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("data-class-distribution_path.png")
plt.clf()

# -------------------- stats per train and test sets ----------------
# Map img_id - dataset index
id_to_idx = map_indices(dataset)
# Load train/test IDs
with open("../checkpoints_path/train_indices.json") as f:
    train_ids = json.load(f)
with open("../checkpoints_path/test_indices.json") as f:
    test_ids = json.load(f)
with open("../checkpoints_path/validation_indices.json") as f:
    val_ids = json.load(f)

# Map IDs back to dataset indices
train_indices = [id_to_idx[i] for i in train_ids]
test_indices = [id_to_idx[i] for i in test_ids]
val_indices = [id_to_idx[i] for i in val_ids]

# Create Subsets
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)
val_dataset = Subset(dataset, val_indices)

# Count labels in train and test
train_labels = [dataset.targets[i] for i in train_indices]
test_labels = [dataset.targets[i] for i in test_indices]
val_labels = [dataset.targets[i] for i in val_indices]

train_counts = Counter(train_labels)
test_counts = Counter(test_labels)
val_counts = Counter(val_labels)

# Class names
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

train_samples = [train_counts.get(i, 0) for i in range(len(class_names))]
test_samples = [test_counts.get(i, 0) for i in range(len(class_names))]
val_samples = [val_counts.get(i, 0) for i in range(len(class_names))]

x = np.arange(len(class_names))  # label locations
width = 0.25                   # width of the bars

plt.figure(figsize=(12,6))
bar1 = plt.bar(x - width/2, train_samples, width, label="Train", color="skyblue")
bar2 = plt.bar(x + width/2, test_samples, width, label="Test", color="lightcoral")
bar3 = plt.bar(x + 1.5 * width, val_samples, width, label="Validation", color="green")

# Add value labels on top of each bar
for bar in bar1 + bar2 + bar3:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 15, int(yval), ha='center', va='bottom', fontsize=8)


plt.ylabel("Number of samples")
plt.xlabel("Classes")
plt.title("Train vs Test vs Validation sample distribution per class for pathMNIST dataset")
plt.xticks(x, full_names, rotation=45)
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("val-test-train-data-class-distribution_path.png")

import os
import json
import torch
from collections import defaultdict
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
import torch.optim as optim
from architecture.model import build_model
from utils.utils import map_indices, get_transform
from torchvision import datasets, transforms


def unlearn(dataset, images=None, config_path="utils/config.json", idx_to_loc_path="checkpoints/idx_to_loc_train.json"):
    """
    Unlearn a specific image or a bach of images
    Parameters:
    - dataset: torchvision.datasets.ImageFolder with the full dataset loaded.
    - images: list of image IDs to unlearn
    """

    with open(config_path) as f:
        cfg = json.load(f)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = cfg["num_classes"]
    NUM_SLICES = cfg["num_slices"]
    NUM_EPOCHS_PER_SLICE = cfg["num_epochs_per_slice"]
    LEARNING_RATE = cfg["learning_rate"]
    BATCH_SIZE = cfg["batch_size"]

    # Load current image ID - (shard, slice) map
    with open(idx_to_loc_path) as f:
        idx_to_loc = json.load(f)

    # map image id to dataset index
    id_to_idx = map_indices(dataset)

    # determine what to unlearn
    remove_ids = set()
    affected_shards = defaultdict(set)

    if images:  # if images is not None, thus we unlearn just one img for a bach
        for img_id in images:
            if str(img_id) in idx_to_loc:
                shard_k, slice_r, label = idx_to_loc[str(img_id)]
                remove_ids.add(img_id)
                affected_shards[shard_k].add(slice_r)

    if not remove_ids:  # if the id of unlearning data is wrong
        print("No matching data found to unlearn.")
        return

    print(f"Start unlearning {len(remove_ids)} images")

    # for each affected shard, rebuild and retrain slices r to end
    for shard_k, slice_set in affected_shards.items():
        min_r = min(slice_set)  # the slice from which we have to start retraining
        print(f"\n Unlearning from shard {shard_k}, starting at slice {min_r}")

        # rebuild slices
        slices = [[] for _ in range(NUM_SLICES)]
        for img_str, (k, r, l) in idx_to_loc.items():
            img_id = int(img_str)
            if k == shard_k and img_id not in remove_ids:
                slices[r].append(id_to_idx[img_id])

        # create slice datasets
        slice_subsets = [Subset(dataset, idxs) for idxs in slices]

        # rebuild model
        model = build_model("resnet18", num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        # load checkpoint before min_r
        if min_r > 0:
            ckpt_path = f"../checkpoints/shard_{shard_k}/slice_{min_r - 1}.pt"
            model.load_state_dict(torch.load(ckpt_path))
            print(f"Loaded checkpoint {ckpt_path}")
        else:
            print("No earlier checkpoint — training from scratch.")

        # retrain slices min_r to end
        for r in range(min_r, NUM_SLICES):
            loader = DataLoader(slice_subsets[r], batch_size=BATCH_SIZE, shuffle=True)
            model.train()
            for epoch in range(NUM_EPOCHS_PER_SLICE):
                for imgs, labels in loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            # save checkpoint
            out_ckpt = f"checkpoints/shard_{shard_k}/slice_{r}.pt"
            torch.save(model.state_dict(), out_ckpt)
            print(f"Updated checkpoint: {out_ckpt}")

    # ! update idx_to_loc by removing deleted image IDs
    for img_id in remove_ids:
        idx_to_loc.pop(str(img_id), None)

    with open(idx_to_loc_path, "w") as f:
        json.dump(idx_to_loc, f, indent=2)
    print(f"\n Unlearning ended. Updated {idx_to_loc_path}")




# load full dataset
transform = get_transform()
dataset = datasets.ImageFolder("data/HAM10000", transform=transform)

# test nlearn a single image
# print("Test 1")
# unlearn(dataset, images=[32125])
#
# print("Test 2")
# # test unlearn multiple images
# unlearn(dataset, images=[29737, 26419, 33498, 26455])

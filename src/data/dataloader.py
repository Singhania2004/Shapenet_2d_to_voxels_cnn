from collections import Counter

from torch.utils.data import DataLoader, WeightedRandomSampler
from src.data.shapenet_dataset import ShapeNetDataset


def get_dataloader(config, split="train"):
    dataset = ShapeNetDataset(config, split=split)

    sampler = None
    shuffle = (split == "train")

    # Optional: oversample small classes so each class is seen equally often
    if split == "train" and config["data"].get("balanced_sampling", False):
        labels = [s[2] for s in dataset.samples]
        counts = Counter(labels)
        weights = [1.0 / counts[l] for l in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False  # sampler and shuffle are mutually exclusive

    nw = config["data"]["num_workers"]
    return DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
    )

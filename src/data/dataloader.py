from torch.utils.data import DataLoader
from src.data.shapenet_dataset import ShapeNetDataset


def get_dataloader(config, split="train"):
    dataset = ShapeNetDataset(config, split=split)

    return DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=(split == "train"),
        num_workers=config["data"]["num_workers"],
    )
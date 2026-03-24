from torch.utils.data import DataLoader
from src.data.shapenet_dataset import ShapeNetDataset


def get_dataloader(config):
    dataset = ShapeNetDataset(config)

    return DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
    )
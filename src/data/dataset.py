import torch
from torch.utils.data import Dataset
import numpy as np


class DummyShapeNetDataset(Dataset):
    """
    Temporary dataset:
    - Random image (simulates input)
    - Random voxel grid (simulates ground truth)
    """

    def __init__(self, num_samples=1000, image_size=128, voxel_size=32):
        self.num_samples = num_samples
        self.image_size = image_size
        self.voxel_size = voxel_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Fake RGB image
        image = np.random.rand(3, self.image_size, self.image_size).astype(np.float32)

        # Fake voxel grid (binary occupancy)
        voxel = np.random.randint(
            0, 2, (1, self.voxel_size, self.voxel_size, self.voxel_size)
        ).astype(np.float32)

        return torch.tensor(image), torch.tensor(voxel)
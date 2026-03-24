import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from src.utils import binvox_rw


class ShapeNetDataset(Dataset):
    def __init__(self, config):
        self.rendering_path = config["paths"]["rendering_dir"]
        self.voxel_path = config["paths"]["voxel_dir"]
        self.class_id = config["data"]["class_id"]

        self.samples = []

        self.transform = transforms.Compose([
            transforms.Resize((config["data"]["image_size"], config["data"]["image_size"])),
            transforms.ToTensor(),
        ])

        self._build_index()

    def _build_index(self):
        render_class_path = os.path.join(self.rendering_path, self.class_id)
        voxel_class_path = os.path.join(self.voxel_path, self.class_id)

        model_ids = os.listdir(render_class_path)

        for model_id in model_ids:
            render_dir = os.path.join(render_class_path, model_id, "rendering")
            voxel_file = os.path.join(voxel_class_path, model_id, "model.binvox")

            if not os.path.exists(render_dir) or not os.path.exists(voxel_file):
                continue

            image_paths = [
                os.path.join(render_dir, img)
                for img in os.listdir(render_dir)
                if img.endswith(".png")
            ]

            if len(image_paths) == 0:
                continue

            self.samples.append((image_paths, voxel_file))

        print(f"Loaded {len(self.samples)} objects")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_paths, voxel_path = self.samples[idx]

        # 🔥 Random view selection
        img_path = random.choice(image_paths)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Load voxel
        with open(voxel_path, 'rb') as f:
            voxel = binvox_rw.read_as_3d_array(f).data.astype(np.float32)

        voxel = np.expand_dims(voxel, axis=0)  # [1,32,32,32]
        voxel = torch.tensor(voxel)

        return image, voxel
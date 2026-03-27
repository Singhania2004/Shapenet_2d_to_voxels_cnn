import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from src.utils import binvox_rw

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class ShapeNetDataset(Dataset):
    def __init__(self, config, split="train"):
        self.split = split
        self.rendering_path = config["paths"]["rendering_dir"]
        self.voxel_path     = config["paths"]["voxel_dir"]
        self.class_id       = config["data"]["class_id"]
        self.samples = []

        img_size = config["data"]["image_size"]
        # For training: return K views; for val/test: return 1 view
        self.num_views = config["data"].get("num_views", 1) if split == "train" else 1

        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

        self._build_index()

    def _build_index(self):
        render_class_path = os.path.join(self.rendering_path, self.class_id)
        voxel_class_path  = os.path.join(self.voxel_path,     self.class_id)

        model_ids = os.listdir(render_class_path)
        model_ids = [m for m in model_ids
                     if os.path.exists(os.path.join(voxel_class_path, m))]

        random.seed(42)
        random.shuffle(model_ids)

        n = len(model_ids)
        split_map = {
            "train": model_ids[:int(0.8 * n)],
            "val":   model_ids[int(0.8 * n):int(0.9 * n)],
            "test":  model_ids[int(0.9 * n):],
        }

        for model_id in split_map[self.split]:
            render_dir = os.path.join(render_class_path, model_id, "rendering")
            voxel_file = os.path.join(voxel_class_path,  model_id, "model.binvox")

            if not os.path.exists(render_dir) or not os.path.exists(voxel_file):
                continue

            image_paths = [
                os.path.join(render_dir, img)
                for img in os.listdir(render_dir)
                if img.endswith(".png")
            ]
            if not image_paths:
                continue

            self.samples.append((image_paths, voxel_file))

        print(f"{self.split} samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_paths, voxel_path = self.samples[idx]

        if self.num_views == 1:
            # Single view → [3, H, W]
            img = Image.open(random.choice(image_paths)).convert("RGB")
            image = self.transform(img)
        else:
            # K distinct random views → [K, 3, H, W]
            chosen = random.sample(image_paths,
                                   min(self.num_views, len(image_paths)))
            # Pad if fewer views available than requested
            while len(chosen) < self.num_views:
                chosen.append(random.choice(image_paths))
            image = torch.stack([
                self.transform(Image.open(p).convert("RGB"))
                for p in chosen
            ])  # [K, 3, H, W]

        with open(voxel_path, "rb") as f:
            voxel = binvox_rw.read_as_3d_array(f).data.astype(np.float32)

        voxel = torch.tensor(voxel).unsqueeze(0)   # [1, 32, 32, 32]
        return image, voxel
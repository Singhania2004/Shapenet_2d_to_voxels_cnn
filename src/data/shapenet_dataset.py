import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from src.utils import binvox_rw


class ShapeNetDataset(Dataset):
    def __init__(self, config, split="train"):
        self.split = split
        self.rendering_path = config["paths"]["rendering_dir"]
        self.voxel_path = config["paths"]["voxel_dir"]
        self.class_id = config["data"]["class_id"]

        self.samples = []

        # 🔥 NEW: Augmentation + normalization
        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((config["data"]["image_size"], config["data"]["image_size"])),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((config["data"]["image_size"], config["data"]["image_size"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        self._build_index()

    def _build_index(self):
        render_class_path = os.path.join(self.rendering_path, self.class_id)
        voxel_class_path = os.path.join(self.voxel_path, self.class_id)

        model_ids = os.listdir(render_class_path)
        model_ids = [m for m in model_ids if os.path.exists(os.path.join(voxel_class_path, m))]

        random.seed(42)
        random.shuffle(model_ids)

        n = len(model_ids)

        train_ids = model_ids[:int(0.8 * n)]
        val_ids = model_ids[int(0.8 * n):int(0.9 * n)]
        test_ids = model_ids[int(0.9 * n):]

        if self.split == "train":
            selected_ids = train_ids
        elif self.split == "val":
            selected_ids = val_ids
        else:
            selected_ids = test_ids

        for model_id in selected_ids:
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

        print(f"{self.split} samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_paths, voxel_path = self.samples[idx]

        img_path = random.choice(image_paths)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        with open(voxel_path, 'rb') as f:
            voxel = binvox_rw.read_as_3d_array(f).data.astype(np.float32)

        voxel = np.expand_dims(voxel, axis=0)
        voxel = torch.tensor(voxel)

        return image, voxel
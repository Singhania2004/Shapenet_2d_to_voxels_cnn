import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from src.utils import binvox_rw


class ShapeNetDataset(Dataset):
    """
    Multi-class ShapeNet dataset.

    Each item: (image [3,H,W], voxel [1,32,32,32], class_idx int)
    class_idx is the position of the class in config["data"]["classes"].

    Splits (80/10/10) are made PER CLASS so every class is represented
    in train / val / test.
    """

    def __init__(self, config, split="train"):
        self.split = split
        self.rendering_path = config["paths"]["rendering_dir"]
        self.voxel_path = config["paths"]["voxel_dir"]

        self.classes = config["data"]["classes"]          # list of {id, name}
        self.class_ids = [c["id"] for c in self.classes]
        self.class_names = [c["name"] for c in self.classes]

        size = config["data"]["image_size"]
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

        if split == "train":
            aug = [
                transforms.Resize((size, size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
            if config["data"].get("hflip_aug", True):
                aug.append(transforms.RandomHorizontalFlip())
            aug += [transforms.ToTensor(), norm]
            self.transform = transforms.Compose(aug)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                norm,
            ])

        self.samples = []   # (image_paths, voxel_file, class_idx)
        self._build_index()

    def _build_index(self):
        for class_idx, class_id in enumerate(self.class_ids):
            render_class_path = os.path.join(self.rendering_path, class_id)
            voxel_class_path = os.path.join(self.voxel_path, class_id)

            if not os.path.isdir(render_class_path) or not os.path.isdir(voxel_class_path):
                print(f"[WARN] missing folder for class {class_id} "
                      f"({self.class_names[class_idx]}), skipping")
                continue

            model_ids = os.listdir(render_class_path)
            model_ids = [m for m in model_ids
                         if os.path.exists(os.path.join(voxel_class_path, m))]

            # Same seed + same procedure as the original single-class code, so the
            # chair split is identical to the one you trained on before
            # (important if you fine-tune from the chair checkpoint: no test leakage).
            # Do NOT sort model_ids, or the chair split will change.
            rng = random.Random(42)
            rng.shuffle(model_ids)

            n = len(model_ids)
            if self.split == "train":
                selected = model_ids[:int(0.8 * n)]
            elif self.split == "val":
                selected = model_ids[int(0.8 * n):int(0.9 * n)]
            else:
                selected = model_ids[int(0.9 * n):]

            count = 0
            for model_id in selected:
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

                self.samples.append((image_paths, voxel_file, class_idx))
                count += 1

            print(f"[{self.split}] {self.class_names[class_idx]:<11} {count}")

        print(f"[{self.split}] total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_paths, voxel_path, class_idx = self.samples[idx]

        img_path = random.choice(image_paths)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        with open(voxel_path, "rb") as f:
            voxel = binvox_rw.read_as_3d_array(f).data.astype(np.float32)

        voxel = torch.from_numpy(np.expand_dims(voxel, axis=0))

        return image, voxel, class_idx

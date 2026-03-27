import os
import torch
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

from src.utils.config import load_config
from src.data.dataloader import get_dataloader
from src.models.model import ReconstructionModel
from src.data.shapenet_dataset import IMAGENET_MEAN, IMAGENET_STD
from train import compute_iou


def main():
    config = load_config("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # For testing, we load the raw dataset to manually fetch ALL views
    test_loader = get_dataloader(config, "test")
    dataset = test_loader.dataset

    model = ReconstructionModel(config).to(device)
    model.load_state_dict(torch.load("outputs/model_best.pth",
                                     map_location=device))
    model.eval()

    total_iou = 0.0

    # Test-time transform (no augmentation)
    transform = transforms.Compose([
        transforms.Resize((config["data"]["image_size"],
                           config["data"]["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    print("Running Multi-View Test Aggregation...")

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            image_paths, voxel_path = dataset.samples[i]

            # ── Load Voxel Ground Truth ──
            # (dataset.__getitem__ normally picks 1 random view, we bypass it)
            _, voxel_gt = dataset[i]
            voxel_gt = voxel_gt.unsqueeze(0).to(device)  # [1, 1, 32, 32, 32]

            # ── Load ALL available views for this sample ──
            all_views = []
            for path in image_paths:
                img = Image.open(path).convert("RGB")
                all_views.append(transform(img))

            if not all_views:
                continue

            # Stack into [1, N, 3, H, W] to feed into multi-view model
            # The model will internally encode all N views and average their features
            images = torch.stack(all_views).unsqueeze(0).to(device)

            # ── Predict and compute IoU ──
            preds = model(images)
            iou = compute_iou(preds, voxel_gt)

            total_iou += iou

    final_iou = total_iou / len(dataset)
    print(f"\nFinal Test IoU (All-Views Aggregated): {final_iou:.4f}")


if __name__ == "__main__":
    main()
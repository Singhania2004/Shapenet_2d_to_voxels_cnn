import os
import torch
from src.utils.config import load_config
from src.data.dataloader import get_dataloader
from src.models.model import ReconstructionModel


def compute_iou_per_sample(pred, target, threshold=0.4):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3, 4))
    union = ((pred + target) > 0).float().sum(dim=(1, 2, 3, 4))
    return intersection / (union + 1e-6)


def main():
    config = load_config("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = [c["name"] for c in config["data"]["classes"]]
    num_classes = len(class_names)

    test_loader = get_dataloader(config, "test")

    model = ReconstructionModel(config).to(device)
    ckpt = os.path.join(config["paths"]["output_dir"], "model_best.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    cls_sum = torch.zeros(num_classes)
    cls_cnt = torch.zeros(num_classes)

    with torch.no_grad():
        for images, voxels, labels in test_loader:
            images, voxels = images.to(device), voxels.to(device)
            ious = compute_iou_per_sample(model(images), voxels, threshold=0.4).cpu()
            cls_sum.index_add_(0, labels, ious)
            cls_cnt.index_add_(0, labels, torch.ones_like(ious))

    per_class = cls_sum / cls_cnt.clamp(min=1)

    print("\nPer-class Test IoU")
    print("-" * 32)
    for name, iou, n in zip(class_names, per_class.tolist(), cls_cnt.int().tolist()):
        print(f"{name:<12} {iou:.4f}   (n={n})")
    print("-" * 32)
    print(f"Mean over samples : {(cls_sum.sum() / cls_cnt.sum()).item():.4f}")
    print(f"Mean over classes : {per_class.mean().item():.4f}")


if __name__ == "__main__":
    main()

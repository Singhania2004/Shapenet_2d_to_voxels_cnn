import torch
from src.utils.config import load_config
from src.data.dataloader import get_dataloader
from src.models.model import ReconstructionModel


def compute_iou_per_sample(pred, target, threshold=0.4):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    intersection = (pred * target).sum(dim=(1,2,3,4))
    union = ((pred + target) > 0).float().sum(dim=(1,2,3,4))

    return (intersection / (union + 1e-6))


def main():
    config = load_config("configs/config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = get_dataloader(config, "test")

    model = ReconstructionModel(config).to(device)
    model.load_state_dict(torch.load("outputs/model_best.pth", map_location=device))
    model.eval()

    total_iou = 0
    total_samples = 0

    with torch.no_grad():
        for images, voxels in test_loader:
            images = images.to(device)
            voxels = voxels.to(device)

            preds = model(images)

            ious = compute_iou_per_sample(preds, voxels, threshold=0.4)

            total_iou += ious.sum().item()
            total_samples += ious.size(0)

    print("Test IoU:", total_iou / total_samples)


if __name__ == "__main__":
    main()
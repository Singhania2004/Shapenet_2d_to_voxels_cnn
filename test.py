import torch
from src.utils.config import load_config
from src.data.dataloader import get_dataloader
from src.models.model import ReconstructionModel
from train import compute_iou


def main():
    config = load_config("configs/config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = get_dataloader(config, "test")

    model = ReconstructionModel(config).to(device)
    model.load_state_dict(torch.load("outputs/model_best.pth"))
    model.eval()

    total_iou = 0

    with torch.no_grad():
        for images, voxels in test_loader:
            images = images.to(device)
            voxels = voxels.to(device)

            preds = model(images)
            iou = compute_iou(preds, voxels)

            total_iou += iou

    print("Test IoU:", total_iou / len(test_loader))


if __name__ == "__main__":
    main()
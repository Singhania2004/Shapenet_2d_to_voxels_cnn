import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils.config import load_config
from src.data.dataloader import get_dataloader
from src.models.model import ReconstructionModel

def compute_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection

    iou = (intersection / (union + 1e-6)).mean()

    return iou.item()

def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    loss = 1 - ((2. * intersection + smooth) / 
                (pred.sum(dim=1) + target.sum(dim=1) + smooth))

    return loss.mean()


def main():
    freeze_epochs = 5
    best_iou = 0
    
    config = load_config("configs/config.yaml")

    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloader = get_dataloader(config)

    model = ReconstructionModel(config).to(device)

    # ❄️ Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    bce = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    model.train()

    for epoch in range(config["training"]["epochs"]):

        epoch_iou = 0   # ✅ reset every epoch

        if epoch == freeze_epochs:
            print("🔥 Unfreezing encoder...")

            for param in model.encoder.parameters():
                param.requires_grad = True

            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=config["training"]["lr"] * 0.1
            )

            model.train()

        loop = tqdm(dataloader)

        for images, voxels in loop:
            images = images.to(device)
            voxels = voxels.to(device)

            preds = model(images)

            loss = 0.5 * bce(preds, voxels) + 0.5 * dice_loss(preds, voxels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iou = compute_iou(preds, voxels)   # ✅ compute first
            epoch_iou += iou

            loop.set_description(f"Epoch [{epoch+1}]")
            loop.set_postfix(loss=loss.item(), iou=iou)

        epoch_iou /= len(dataloader)

        print(f"Epoch [{epoch+1}] IoU: {epoch_iou:.4f}")

        if epoch_iou > best_iou:
            best_iou = epoch_iou
            torch.save(model.state_dict(), "outputs/model_best.pth")
            print("✅ Saved best model")

    torch.save(model.state_dict(), "outputs/model_final.pth")


if __name__ == "__main__":
    main()
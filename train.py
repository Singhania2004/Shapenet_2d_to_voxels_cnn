import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils.config import load_config
from src.data.dataloader import get_dataloader
from src.models.model import ReconstructionModel

# ✅ NEW IoU FUNCTION
def compute_iou(pred, target, threshold=0.5 ):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    # Summing over all dimensions except batch (D, H, W, C)
    intersection = (pred * target).sum(dim=(1, 2, 3, 4))
    union = ((pred + target) > 0).float().sum(dim=(1, 2, 3, 4))

    return (intersection / (union + 1e-6)).mean().item()

# ✅ NEW DICE LOSS
def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum(dim=(1, 2, 3, 4))
    union = pred.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def main():
    freeze_epochs = 5
    best_val_iou = 0

    config = load_config("configs/config.yaml")

    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # ✅ Loaders
    train_loader = get_dataloader(config, "train")
    val_loader = get_dataloader(config, "val")

    model = ReconstructionModel(config).to(device)

    # ❄️ Freeze encoder initially
    for param in model.encoder.parameters():
        param.requires_grad = False

    # ✅ NEW LOSS INITIALIZATION
    # Increased pos_weight to 5.0 as requested
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))

    def compute_loss(pred, target):
        bce = bce_loss_fn(pred, target)
        dice = dice_loss(pred, target)
        return bce + 0.5 * dice

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    for epoch in range(config["training"]["epochs"]):
        model.train()
        epoch_iou = 0

        # 🔥 Unfreeze encoder
        if epoch == freeze_epochs:
            print("🔥 Unfreezing encoder...")
            for param in model.encoder.parameters():
                param.requires_grad = True

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config["training"]["lr"] * 0.1
            )

        loop = tqdm(train_loader)

        # ================= TRAIN =================
        for images, voxels in loop:
            images = images.to(device)
            voxels = voxels.to(device)

            preds = model(images)

            # ✅ Using the new composite loss
            loss = compute_loss(preds, voxels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iou = compute_iou(preds, voxels)
            epoch_iou += iou

            loop.set_description(f"Epoch [{epoch+1}]")
            loop.set_postfix(loss=loss.item(), iou=iou)

        epoch_iou /= len(train_loader)
        print(f"Train IoU: {epoch_iou:.4f}")

        # ================= VALIDATION =================
        model.eval()
        val_iou = 0

        with torch.no_grad():
            for images, voxels in val_loader:
                images = images.to(device)
                voxels = voxels.to(device)

                preds = model(images)
                iou = compute_iou(preds, voxels)

                val_iou += iou

        val_iou /= len(val_loader)
        print(f"Validation IoU: {val_iou:.4f}")

        # ✅ Save BEST model based on VAL
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), "outputs/model_best.pth")
            print("✅ Saved best model")

    torch.save(model.state_dict(), "outputs/model_final.pth")

if __name__ == "__main__":
    main()
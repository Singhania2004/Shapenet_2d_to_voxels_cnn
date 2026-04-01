import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.config import load_config
from src.data.dataloader import get_dataloader
from src.models.model import ReconstructionModel


# ✅ IoU (threshold = 0.4)
def compute_iou(pred, target, threshold=0.4):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    intersection = (pred * target).sum(dim=(1, 2, 3, 4))
    union = ((pred + target) > 0).float().sum(dim=(1, 2, 3, 4))

    return (intersection / (union + 1e-6)).mean().item()


# ✅ Dice Loss
def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum(dim=(1, 2, 3, 4))
    union = pred.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


# ✅ FIXED Focal Loss (stable)
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)
    loss = alpha * (1 - pt) ** gamma * bce
    return loss.mean()


def main():
    best_val_iou = 0

    config = load_config("configs/config.yaml")

    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    train_loader = get_dataloader(config, "train")
    val_loader = get_dataloader(config, "val")

    model = ReconstructionModel(config).to(device)

    # Loss
    bce_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([5.0]).to(device)
    )

    def compute_loss(pred, target):
        bce = bce_loss_fn(pred, target)
        dice = dice_loss(pred, target)
        focal = focal_loss(pred, target)

        # 🔥 Balanced loss (important)
        return bce + 0.5 * dice + 0.2 * focal

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"]  # recommend 5e-5 in config
    )

    # 🔥 Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=8, gamma=0.5
    )

    for epoch in range(config["training"]["epochs"]):
        model.train()
        epoch_iou = 0

        loop = tqdm(train_loader)

        for images, voxels in loop:
            images = images.to(device)
            voxels = voxels.to(device)

            preds = model(images)

            # 🔥 Warm-up (first 2 epochs → only BCE)
            if epoch < 2:
                loss = bce_loss_fn(preds, voxels)
            else:
                loss = compute_loss(preds, voxels)

            optimizer.zero_grad()
            loss.backward()

            # 🔥 Gradient clipping (stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), "outputs/model_best.pth")
            print("✅ Saved best model")

        # 🔥 Step scheduler
        scheduler.step()

    torch.save(model.state_dict(), "outputs/model_final.pth")


if __name__ == "__main__":
    main()
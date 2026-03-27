import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.config import load_config
from src.data.dataloader import get_dataloader
from src.models.model import ReconstructionModel


# ─────────────────────────── Metrics ────────────────────────────

def compute_iou(pred_logits, target, threshold=0.5):
    pred = torch.sigmoid(pred_logits)
    pred = (pred > threshold).float()

    pred   = pred.reshape(pred.size(0), -1)
    target = target.reshape(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection

    return (intersection / (union + 1e-6)).mean().item()


# ─────────────────────────── Losses ─────────────────────────────

def focal_loss(pred_logits, target, gamma=2.0, alpha=0.85):
    """
    Binary focal loss.
    gamma=2 heavily down-weights easy-to-classify empty voxels.
    alpha=0.85 strongly up-weights the rare filled voxels (legs/arms).
    """
    bce = F.binary_cross_entropy_with_logits(
        pred_logits, target, reduction="none")
    p_t = torch.exp(-bce)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    fl = alpha_t * (1 - p_t) ** gamma * bce
    return fl.mean()


def dice_loss(pred_logits, target, smooth=1.0):
    pred = torch.sigmoid(pred_logits)
    pred   = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    loss = 1 - (2.0 * intersection + smooth) / (
        pred.sum(dim=1) + target.sum(dim=1) + smooth)
    return loss.mean()


def combined_loss(pred_logits, target):
    """Focal (0.6) + Dice (0.4)"""
    return 0.6 * focal_loss(pred_logits, target) + \
           0.4 * dice_loss(pred_logits, target)


# ─────────────────────────── Main ───────────────────────────────

def main():
    freeze_epochs = 3  # Less freeze time, encoder needs to learn multiview quickly
    best_val_iou  = 0.0

    config = load_config("configs/config.yaml")
    epochs = config["training"]["epochs"]

    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train returns [B, K, 3, H, W], Val returns [B, 3, H, W]
    train_loader = get_dataloader(config, "train")
    val_loader   = get_dataloader(config, "val")

    model = ReconstructionModel(config).to(device)

    # ❄️ Freeze encoder initially
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["lr"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    for epoch in range(epochs):

        # 🔥 Unfreeze encoder
        if epoch == freeze_epochs:
            print("🔥 Unfreezing encoder...")
            for param in model.encoder.parameters():
                param.requires_grad = True

            # Drop LR slightly when unfreezing full network
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config["training"]["lr"] * 0.1
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - freeze_epochs, eta_min=1e-6)

        # ─── Train ───────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        epoch_iou  = 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for images, voxels in loop:
            # images is [B, K, 3, 128, 128] for train
            images = images.to(device)
            voxels = voxels.to(device)

            preds = model(images)
            loss  = combined_loss(preds, voxels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            iou = compute_iou(preds, voxels)
            epoch_loss += loss.item()
            epoch_iou  += iou

            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                iou=f"{iou:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

        scheduler.step()

        train_iou  = epoch_iou  / len(train_loader)
        train_loss = epoch_loss / len(train_loader)
        print(f"Train  — Loss: {train_loss:.4f} | IoU: {train_iou:.4f}")

        # ─── Validation ──────────────────────────────────────────
        model.eval()
        val_iou = 0.0

        with torch.no_grad():
            for images, voxels in val_loader:
                # val images is single-view [B, 3, 128, 128]
                images = images.to(device)
                voxels = voxels.to(device)

                preds  = model(images)
                val_iou += compute_iou(preds, voxels)

        val_iou /= len(val_loader)
        print(f"Val    — IoU: {val_iou:.4f}")

        # ─── Save best checkpoint ─────────────────────────────────
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), "outputs/model_best.pth")
            print(f"✅ Saved best model (val IoU: {best_val_iou:.4f})")

    torch.save(model.state_dict(), "outputs/model_final.pth")
    print(f"\nTraining complete. Best Val IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    main()
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.config import load_config
from src.data.dataloader import get_dataloader
from src.models.model import ReconstructionModel


def iou_per_sample(pred, target, threshold=0.4):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3, 4))
    union = ((pred + target) > 0).float().sum(dim=(1, 2, 3, 4))
    return intersection / (union + 1e-6)


def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1, 2, 3, 4))
    union = pred.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    pt = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    """Returns (overall mean IoU, per-class IoU tensor)."""
    model.eval()
    cls_sum = torch.zeros(num_classes)
    cls_cnt = torch.zeros(num_classes)

    for images, voxels, labels in loader:
        images, voxels = images.to(device), voxels.to(device)
        ious = iou_per_sample(model(images), voxels).cpu()
        cls_sum.index_add_(0, labels, ious)
        cls_cnt.index_add_(0, labels, torch.ones_like(ious))

    per_class = cls_sum / cls_cnt.clamp(min=1)
    overall = cls_sum.sum() / cls_cnt.sum().clamp(min=1)
    return overall.item(), per_class


def main():
    config = load_config("configs/config.yaml")
    tcfg = config["training"]
    out_dir = config["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    class_names = [c["name"] for c in config["data"]["classes"]]
    num_classes = len(class_names)

    device = torch.device(tcfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = get_dataloader(config, "train")
    val_loader = get_dataloader(config, "val")

    model = ReconstructionModel(config).to(device)

    # Optional: start from the chair-only checkpoint
    if tcfg.get("init_from"):
        model.load_state_dict(torch.load(tcfg["init_from"], map_location=device))
        print(f"Initialised from {tcfg['init_from']}")

    bce_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([tcfg.get("pos_weight", 5.0)]).to(device)
    )

    def compute_loss(pred, target):
        return (bce_loss_fn(pred, target)
                + 0.5 * dice_loss(pred, target)
                + 0.2 * focal_loss(pred, target))

    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    warmup = tcfg.get("warmup_epochs", 2)
    best_val_iou = 0.0

    for epoch in range(tcfg["epochs"]):
        model.train()
        epoch_iou = 0.0

        loop = tqdm(train_loader)
        for images, voxels, _ in loop:
            images, voxels = images.to(device), voxels.to(device)

            preds = model(images)
            loss = bce_loss_fn(preds, voxels) if epoch < warmup else compute_loss(preds, voxels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            iou = iou_per_sample(preds.detach(), voxels).mean().item()
            epoch_iou += iou
            loop.set_description(f"Epoch [{epoch + 1}/{tcfg['epochs']}]")
            loop.set_postfix(loss=loss.item(), iou=iou)

        print(f"Train IoU: {epoch_iou / len(train_loader):.4f}")

        # ---------------- validation ----------------
        val_iou, per_class = evaluate(model, val_loader, device, num_classes)
        print(f"Validation IoU (all): {val_iou:.4f}")
        print("  " + " | ".join(f"{n}: {v:.3f}" for n, v in zip(class_names, per_class.tolist())))

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pth"))
            print("✅ Saved best model")

        scheduler.step()

    torch.save(model.state_dict(), os.path.join(out_dir, "model_final.pth"))


if __name__ == "__main__":
    main()

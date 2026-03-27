import torch
import yaml
from src.models.model import ReconstructionModel

config = yaml.safe_load(open("configs/config.yaml"))
m = ReconstructionModel(config).cuda()

# Test 1: Multi-view Input [B, K, 3, 128, 128]
print("Running Multi-View Forward...")
x_multi = torch.randn(2, 3, 3, 128, 128).cuda()
out_multi = m(x_multi)
print("  Output shape (K=3):", out_multi.shape)
assert out_multi.shape == (2, 1, 32, 32, 32), "Multi-view shape mismatch"

# Test 2: Single-view Input [B, 3, 128, 128]
print("Running Single-View Forward...")
x_single = torch.randn(2, 3, 128, 128).cuda()
out_single = m(x_single)
print("  Output shape (K=1):", out_single.shape)
assert out_single.shape == (2, 1, 32, 32, 32), "Single-view shape mismatch"

total     = sum(p.numel() for p in m.parameters())
trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"\nTotal params:     {total:,}")
print(f"Trainable params: {trainable:,}")
print("\nMulti-View Shape Checks PASSED ✅")

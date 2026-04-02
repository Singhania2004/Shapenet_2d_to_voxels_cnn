"""
Voxel Reconstruction Web App — Flask Backend
Run: python app.py
Then open: http://localhost:5000
Place your model_best.pth at: outputs/model_best.pth
"""

import io
import os
import sys
import json
import base64
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── path setup ──────────────────────────────────────────────────────────────
# Adjust this to point at the root of your shapenet project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.models.model import ReconstructionModel  # noqa: E402

# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(PROJECT_ROOT, "outputs", "model_best.pth")
LATENT_DIM   = 1024
THRESHOLD    = 0.4          # default sigmoid threshold
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ── Lazy-load model once ──────────────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        config = {"model": {"latent_dim": LATENT_DIM}}
        _model = ReconstructionModel(config).to(DEVICE)
        _model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE)
        )
        _model.eval()
        print(f"[✓] Model loaded from {MODEL_PATH} on {DEVICE}")
    return _model


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    threshold = float(request.form.get("threshold", THRESHOLD))
    file      = request.files["image"]

    # ── preprocess ────────────────────────────────────────────────────────────
    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    # base64-encode original image to send back for display
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)          # [1,3,128,128]

    # ── inference ─────────────────────────────────────────────────────────────
    model = get_model()
    with torch.no_grad():
        logits     = model(tensor)                            # [1,1,32,32,32]
        prob_vol   = torch.sigmoid(logits).cpu().numpy()      # [1,1,32,32,32]

    prob_vol = prob_vol[0, 0]                                 # [32,32,32]

    # ── extract occupied voxels ───────────────────────────────────────────────
    # rotate to match notebook orientation
    # vis_vol = np.rot90(prob_vol, k=1, axes=(1, 2))
    occupied = np.argwhere(prob_vol > threshold)              # [N,3]

    # occupancy stats
    num_occupied = int((prob_vol > threshold).sum())
    total_voxels = int(prob_vol.size)

    # IoU at multiple thresholds (relative, for UI display)
    iou_stats = {}
    for t in [0.3, 0.4, 0.5]:
        binary = (prob_vol > t)
        iou_stats[str(t)] = round(float(binary.sum()) / total_voxels, 4)

    payload = {
        "voxels": occupied.tolist(),          # list of [x,y,z] triples
        "grid_size": int(prob_vol.shape[0]),
        "num_occupied": num_occupied,
        "total_voxels": total_voxels,
        "threshold": threshold,
        "iou_stats": iou_stats,
        "image_b64": img_b64,
    }

    return jsonify(payload)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)})


if __name__ == "__main__":
    print(f"[*] Starting server — model will load on first request")
    print(f"[*] Open  http://localhost:5000  in your browser")
    app.run(host="0.0.0.0", port=5000, debug=False)

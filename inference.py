import torch
from PIL import Image
import torchvision.transforms as transforms

from src.utils.config import load_config
from src.models.model import ReconstructionModel
from src.utils.voxel_to_mesh import voxel_to_mesh, save_mesh


def main():
    config = load_config("configs/config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ReconstructionModel(config).to(device)

    # 🔥 Load trained weights (VERY IMPORTANT)
    model.load_state_dict(torch.load("outputs/model_final.pth"))
    model.eval()

    # 🔥 Use REAL image from ShapeNet
    image_path = "src/data/ShapeNetRendering/ShapeNetRendering/03001627/d7db1353551d341f149c35efde9de588/rendering/18.png"

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")   
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        voxel_probs = torch.sigmoid(logits)
        # Threshold at 0.5 to create binary voxel grid
        voxel = (voxel_probs > 0.5).float().cpu().numpy()

    mesh = voxel_to_mesh(voxel)

    save_mesh(mesh, "outputs/output.obj")

    print("3D model saved at outputs/output.obj")


if __name__ == "__main__":
    main()
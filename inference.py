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
    image_path = "src/data/ShapeNetRendering/ShapeNetRendering/03001627/1eab4c4c55b8a0f48162e1d15342e13b/rendering/16.png"

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")   
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        voxel = model(image).cpu().numpy()

    mesh = voxel_to_mesh(voxel)

    save_mesh(mesh, "outputs/output.obj")

    print("3D model saved at outputs/output.obj")


if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image

from models import Generator
from utils import scale, scale_back, show

# Settings
MODEL_PATH = "pokemon.pth"
MODEL_LEGACY = False
REAL_IMAGE_PATH = "00000023.png"

def transform():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Network
    g = Generator(64)
    g = g.to(device)
    g = nn.DataParallel(g)

    # Load Weights
    if (MODEL_LEGACY):
        g.load_state_dict(torch.load(MODEL_PATH))
    else:
        g.load_state_dict(torch.load(MODEL_PATH)["model_state_dict"])

    g = g.to(device)

    # Load Image and convert to PyTorch tensor
    real_image = Image.open(REAL_IMAGE_PATH).convert("RGB")
    transform = transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    real_tensor = transform(real_image).unsqueeze(0)

    with torch.no_grad():
        # Scale - Forward Pass - Scale Back
        real_tensor = scale(real_tensor)
        fake_tensor = g(real_tensor)
        fake_tensor = scale_back(fake_tensor)
        fake_tensor = fake_tensor.squeeze(0)

        # Tensor to Numpy Array
        fake_image = fake_tensor.cpu().numpy().transpose(1,2,0)
        show(fake_image)

transform()
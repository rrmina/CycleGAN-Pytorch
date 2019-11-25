import torch
import torchvision
from models import Generator, Discriminator
from utils import scale, scale_back, concat_images

def evaluate(G_net, real_images):
    G_net.eval()
    generated = G_net(real_images)
    return generated
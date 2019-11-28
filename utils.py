import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms

import cv2
import numpy as np

def prepare_loader(path, name, transform, batch_size=1, shuffle=True, num_workers=1):
    # Path
    train_folder = path + "train" + name
    test_folder = path + "test" + name

    # Dataset
    train_dataset = datasets.ImageFolder(train_folder, transform=transform)
    test_dataset = datasets.ImageFolder(test_folder, transform=transform)    

    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader, test_loader

# Merge Tensors into Adjacent numpy arrays!
def concat_images(tensor):
    image_tensor = torchvision.utils.make_grid(tensor)          # Merge Tensor [B, C, H, W] -> [C, H, W*B]
    image_tensor = scale_back(image_tensor)                     # [-1, 1] -> [0, 1]
    concat_tensor = image_tensor.clone().detach().cpu().numpy() # Torch Tensor to Numpy Array
    images = concat_tensor.transpose(1,2,0)                     # [C, H, W] -> [H, W, C]
    return images

# Expects a torch tensor. Please do not pass a numpy tensor.
def show_tensor(tensor, title=""):
    image = tensor.clone().detach().cpu().numpy()
    image = image.transpose(1,2,0)
    show(image, title)

# Expects a numpy array
def show(image, title=""):
    fig = plt.figure(figsize=(10,10))
    plt.title(title)
    plt.imshow(image)
    plt.show()

# Plot Loss Curves
def plot_loss(loss_hist):
    fig = plt.figure(figsize=(10,10))
    plt.plot(loss_hist["Gxy"], label="Gxy")
    plt.plot(loss_hist["Gyx"], label="Gyx")
    plt.plot(loss_hist["Dxy"], label="Dxy")
    plt.plot(loss_hist["Dyx"], label="Dyx")
    plt.plot(loss_hist["cycle"], label="cycle")
    plt.title("Training Losses")
    plt.legend()
    plt.show()

# Save Image
def saveimg(image, savepath):
    plt.imsave(savepath, image)

# Output of generator is Tanh! so we need to scale real images accordingly
def scale(tensor, mini=-1, maxi=1):
    return tensor * (maxi-mini) + mini

# Outputs need to be scaled back from [mini, maxi] to [0, 1]
def scale_back(tensor, mini=-1, maxi=1):
    return (tensor-mini)/(maxi-mini)
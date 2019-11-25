import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

import cv2
import numpy as np

def prepare_loader(path, name, transform, batch_size=1, shuffle=True):
    # Path
    train_folder = path + "train" + name
    test_folder = path + "test" + name

    # Dataset
    train_dataset = datasets.ImageFolder(train_folder, transform=transform)
    test_dataset = datasets.ImageFolder(test_folder, transform=transform)    

    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader

def ttoi(tensor):
    img = tensor.cpu().numpy()
    return img

def show(img):
    plt.figure(figsize=(30, 30))
    plt.imshow(img)
    plt.show()

def saveimg(img, image_path):
    img = img.clip(0,1)

    # Transpose from [C, H, W] -> [H, W, C]
    img = img.transpose(1,2,0)
    #cv2.imwrite(image_path, img)
    plt.imsave(image_path, img)

def merge_images(original, generated):
    # Check sizes
    bs, c, h, w = original.shape
    num_rows = 2
    num_cols = bs

    # Placeholder Image
    placeholder = np.empty([3, num_rows*h, num_cols*w])
    
    # Reshape
    for i in range(num_rows):
        for j in range(num_cols):
            if (i == 0):
                placeholder[:, i*h: (i+1)*h, j*w: (j+1)*w] = original[j]
            else:
                placeholder[:, i*h: (i+1)*h, j*w: (j+1)*w] = generated[j]

    return placeholder.transpose(1, 2, 0)

def savesamples(original, generated):
    image = merge_images(original, generated)

def concatenate_images(original, generated, H, W):
    # Compute necessary dimension
    batch_size = original.shape[0]
    pixel_row = batch_size * W
    pixel_col = 2 * H

    # Placeholder Image
    placeholder = np.empty([3, pixel_col, pixel_row])

    # Reshape
    for i in range(2):
        for j in range(batch_size):
            if (i == 0):
                placeholder[ :, i*H : (i+1)*H, j*W : (j+1)*W ] = original[j]
            else:
                placeholder[ :, i*H : (i+1)*H, j*W : (j+1)*W ] = generated[j]

    return placeholder
            
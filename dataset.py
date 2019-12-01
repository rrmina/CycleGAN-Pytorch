
import os
import shutil
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Script to loader paired images for aligned dataset
# Used for pokemon dataset
# Adapted from https://discuss.pytorch.org/t/how-can-i-read-a-image-pair-at-the-same-time/23151/2?u=rusty
class PairedDataset(Dataset):
    def __init__(self, image1_paths, image2_paths, transform=None):
        
        A = []
        B = []
        for folder in os.listdir(image1_paths):
            A_folder = os.path.join(image1_paths, folder)
            B_folder = os.path.join(image2_paths, folder)
            
            for image in os.listdir(A_folder):
                A.append(os.path.join(A_folder, image))
                B.append(os.path.join(B_folder, image))
        
        self.image1_paths = A
        self.image2_paths = B
        self.transform = transform
        
    def __getitem__(self, index):
        img1 = Image.open(self.image1_paths[index]).convert("RGB")
        img2 = Image.open(self.image2_paths[index]).convert("RGB")
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2
    
    def __len__(self):
        return len(self.image1_paths)

# Scripts to preprocess dataset Taesung Park's repository
# Turn train and test folder into subfolders
# torchvision.datasets.ImageFolder only accepts folder with subfolders
def prepare_dataset_and_folder(DATASET_PATH, folder_names=[]):
    prepare_dataset(DATASET_PATH)
    prepare_folder(folder_names)

def prepare_dataset(DATASET_PATH):
    
    folder_names = os.listdir(DATASET_PATH) # testA, testB, trainA, trainB

    # Rename Original Folders
    os.rename(DATASET_PATH + folder_names[0], DATASET_PATH + "A") # testA
    os.rename(DATASET_PATH + folder_names[1], DATASET_PATH + "B") # testB
    os.rename(DATASET_PATH + folder_names[2], DATASET_PATH + "C") # trainA
    os.rename(DATASET_PATH + folder_names[3], DATASET_PATH + "D") # trainB

    # Make original folder names
    for name in folder_names:
        os.mkdir(DATASET_PATH + name)

    # Original Folders become subfolder
    shutil.move(DATASET_PATH + "A", DATASET_PATH + folder_names[0]) # testA/A
    shutil.move(DATASET_PATH + "B", DATASET_PATH + folder_names[1]) # testB/B
    shutil.move(DATASET_PATH + "C", DATASET_PATH + folder_names[2]) # trainA/C
    shutil.move(DATASET_PATH + "D", DATASET_PATH + folder_names[3]) # trainB/D

    # Rename subfolder for brevity
    os.rename(DATASET_PATH + folder_names[0] + "/A", DATASET_PATH + folder_names[0] + "/images") # testA/images
    os.rename(DATASET_PATH + folder_names[1] + "/B", DATASET_PATH + folder_names[1] + "/images") # testB/images
    os.rename(DATASET_PATH + folder_names[2] + "/C", DATASET_PATH + folder_names[2] + "/images") # trainA/images
    os.rename(DATASET_PATH + folder_names[3] + "/D", DATASET_PATH + folder_names[3] + "/images") # trainB/images

def prepare_folder(folder_names=[]):
    for name in folder_names:
        if (os.path.exists(name) == False):
            os.mkdir(name)
# Script to preprocess dataset Taesung Park's repository
# Turn train and test folder into subfolders
# torchvision.datasets.ImageFolder only accepts folder with subfolders

import os
import shutil

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
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools

import models
import utils
import time

import losses
from losses import real_loss, fake_loss, cycle_loss
from dataset import prepare_dataset_and_folder
import generate

# Global Settings
TRAIN_IMAGE_SIZE = 128
DATASET_PATH = "summer2winter_yosemite/" # summer2winter_yosemite
X_CLASS = "A"
Y_CLASS = "B"
IMAGE_SAVE_FOLDER = "results/"
MODEL_SAVE_FOLDER = "pretrained_models/"
SAVE_MODEL_EVERY = 10

BATCH_SIZE = 4
SEED = 35

# Optimizer Settings
LR = 2e-4
START_LR_DECAY = 100
BETA_1 = 0.5
BETA_2 = 0.999
NUM_EPOCHS = 2
CYCLE_WEIGHT = 10

def train():
    # Seeds
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    # Prepare dataset to make it usable with ImageFolder. Please only do this once
    # Uncomment this when you encounter "RuntimeError: Found 0 files in subfolders of:"
    # prepare_dataset_and_folder(DATASET_PATH, [IMAGE_SAVE_FOLDER, MODEL_SAVE_FOLDER])

    # Tranform, Dataset, DataLoaders
    transform = transforms.Compose([
        transforms.Resize(TRAIN_IMAGE_SIZE),
        transforms.CenterCrop(TRAIN_IMAGE_SIZE),
        transforms.ToTensor()
    ])
    x_trainloader, x_testloader = utils.prepare_loader(DATASET_PATH, X_CLASS, transform=transform, batch_size=BATCH_SIZE, shuffle=True)
    y_trainloader, y_testloader = utils.prepare_loader(DATASET_PATH, Y_CLASS, transform=transform, batch_size=BATCH_SIZE, shuffle=True)

    # [Iterators]: We need iterators for DataLoader because we 
    # are fetching training samples from 2 or more DataLoaders
    x_trainiter, x_testiter = iter(x_trainloader), iter(x_testloader)
    y_trainiter, y_testiter = iter(y_trainloader), iter(y_testloader)

    # Load Networks
    Gxy = models.Generator(64).to(device)
    Gyx = models.Generator(64).to(device)
    Dxy = models.Discriminator(64).to(device)
    Dyx = models.Discriminator(64).to(device)

    # Initialize weights with Normal(0, 0.02)
    Gxy.apply(models.init_weights)
    Gyx.apply(models.init_weights)
    Dxy.apply(models.init_weights)
    Dyx.apply(models.init_weights)

    # Optimizers
    # Concatenate Generator ParamsSee Training Loop for explanation
    # Reference: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/cycle_gan_model.py#L94             
    G_optim = optim.Adam(itertools.chain(Gxy.parameters(), Gyx.parameters()), lr=LR, betas=[BETA_1, BETA_2]) 
    Dxy_optim = optim.Adam(Dxy.parameters(), lr=LR, betas=[BETA_1, BETA_2])
    Dyx_optim = optim.Adam(Dyx.parameters(), lr=LR, betas=[BETA_1, BETA_2])

    # LR Scheduler
    # Reference: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L51
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - START_LR_DECAY)/(NUM_EPOCHS - START_LR_DECAY)
        return lr_l
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(G_optim, lr_lambda=lambda_rule)
    lr_scheduler_Dxy = torch.optim.lr_scheduler.LambdaLR(Dxy_optim, lr_lambda=lambda_rule)
    lr_scheduler_Dyx = torch.optim.lr_scheduler.LambdaLR(Dyx_optim, lr_lambda=lambda_rule)

    # Some Helper Functions!    
    # Output of generator is Tanh! so we need to scale real images accordingly
    def scale(tensor, mini=-1, maxi=1):
        return tensor * (maxi-mini) + mini

    # Fixed test samples
    fixed_X, _ = next(x_testiter)
    fixed_X = fixed_X.to(device)   
    fixed_Y, _ = next(y_testiter)
    fixed_Y = fixed_Y.to(device)

    # Loss History for the Whole Training Process
    loss_hist = losses.createLogger(["Gxy", "Gyx", "Dxy", "Dyx", "cycle"])

    # Number of batches
    iter_per_epoch = min(len(x_trainiter), len(y_trainiter))
    print("There are {} batches per epoch".format(iter_per_epoch))

    for epoch in range(1, NUM_EPOCHS+1):
        print("========Epoch {}/{}========".format(epoch, NUM_EPOCHS))
        start_time = time.time()
        # Loss Logger for the Current Batch
        curr_loss_hist = losses.createLogger(["Gxy", "Gyx", "Dxy", "Dyx", "cycle"])

        # Reset Iterators every epoch, otherwise, we'll have unequal batch sizes 
        # or worse, reach the end of iterator and get a Stop Iteration Error
        x_trainiter = iter(x_trainloader)
        y_trainiter = iter(y_trainloader)

        for i in range(1, iter_per_epoch):

            # Get current batches
            x_real, _ = next(x_trainiter) 
            x_real = scale(x_real)
            x_real = x_real.to(device)
            
            y_real, _ = next(y_trainiter)
            y_real = scale(y_real)
            y_real = y_real.to(device)
            
            # ========= Discriminator ==========
            # In training the discriminators, we fix the generators' parameters
            # It is alright to train both discriminators seperately because
            # their forward pass don't share any parameters with each other

            # Discriminator Y -> X Adversarial Loss
            Dyx_optim.zero_grad()                       # Zero-out Gradients
            Dyx_real_out = Dyx(x_real)                  # Dyx Forward Pass
            Dyx_real_loss = real_loss(Dyx_real_out)     # Dyx Real Loss 
            Dyx_fake_out = Dyx(Gyx(y_real))             # Gyx produces fake-x images
            Dyx_fake_loss = fake_loss(Dyx_fake_out)     # Dyx Fake Loss
            Dyx_loss = Dyx_real_loss + Dyx_fake_loss    # Dyx Total Loss
            Dyx_loss.backward()                         # Dyx Backprop
            Dyx_optim.step()                            # Dyx Gradient Descent
            
            # Discriminator X -> Y Adversarial Loss
            Dxy_optim.zero_grad()                       # Zero-out Gradients
            Dxy_real_out = Dxy(y_real)                  # Dxy Forward Pass
            Dxy_real_loss = real_loss(Dxy_real_out)     # Dxy Real Loss
            Dxy_fake_out = Dxy(Gxy(x_real))             # Gxy produces fake y-images
            Dxy_fake_loss = fake_loss(Dxy_fake_out)     # Dxy Fake Loss
            Dxy_loss = Dxy_real_loss + Dxy_fake_loss    # Dxy Total Loss
            Dxy_loss.backward()                         # Dxy Backprop
            Dxy_optim.step()                            # Dxy Gradient Descent

            # ============= Generator ==============
            # Similar to training discriminator networks, in training 
            # generator networks, we fix discriminator networks.
            # However, cycle consistency prohibits us 
            # from training generators seperately.

            # Generator X -> Y Adversarial Loss
            G_optim.zero_grad()                         # Zero-out Gradients
            Gxy_out = Gxy(x_real)                       # Gxy Forward Pass : generates fake-y images
            D_Gxy_out = Dxy(Gxy_out)                    # Gxy -> Dxy Forward Pass
            Gxy_loss = real_loss(D_Gxy_out)             # Gxy Real Loss
            
            # Generator Y -> X Adversarial Loss
            Gyx_out = Gyx(y_real)                       # Gyx Forward Pass : generates fake-x images
            D_Gyx_out = Dyx(Gyx_out)                    # Gyx -> Dyx Forward Pass    
            Gyx_loss = real_loss(D_Gyx_out)             # Gyx Real Loss
            
            # Cycle Consistency Loss
            yxy = Gxy(Gyx_out)                          # Reconstruct Y
            yxy_cycle_loss = cycle_loss(yxy, y_real)    # Y-X-Y L1 Cycle Reconstruction Loss
            xyx = Gyx(Gxy_out)                          # Reconstruct X
            xyx_cycle_loss = cycle_loss(xyx, x_real)    # X-Y-X L1 Cycle Reconstruction Loss
            G_cycle_loss = CYCLE_WEIGHT * (yxy_cycle_loss + xyx_cycle_loss)
            
            # Generator Total Loss
            G_loss = Gxy_loss + Gyx_loss + G_cycle_loss
            G_loss.backward()
            G_optim.step()

            # Record Losses
            curr_loss_hist = losses.updateEpochLogger(curr_loss_hist, [Gxy_loss, Gyx_loss, Dxy_loss, Dxy_loss, G_cycle_loss])

        # Learning Rate Scheduler Step
        lr_scheduler_G.step()
        lr_scheduler_Dxy.step()
        lr_scheduler_Dyx.step()

        # Record and Print Losses
        print("Dxy: {} Dyx: {} G: {} Cycle: {}".format(Dxy_loss.item(), Dyx_loss.item(), G_loss.item(), G_cycle_loss.item()))
        print("Time Elapsed: {}".format(time.time() - start_time))
        loss_hist = losses.updateGlobalLogger(loss_hist, curr_loss_hist)

        # Generate Fake Images
        Gxy.eval()
        Gyx.eval()
        with torch.no_grad():
            # Generate Fake X Images  
            x_tensor = generate.evaluate(Gyx, fixed_Y)  # Generate Image Tensor
            x_images = utils.concat_images(x_tensor)    # Merge Image Tensors -> Numpy Array
            save_path = IMAGE_SAVE_FOLDER + DATASET_PATH[:-1] + "X" + str(epoch) + ".png"
            utils.saveimg(x_images, save_path)

            # Generate Fake Y Images
            y_tensor = generate.evaluate(Gxy, fixed_X)  # Generate Image Tensor
            y_images = utils.concat_images(y_tensor)    # Merge Image Tensors -> Numpy Array
            save_path = IMAGE_SAVE_FOLDER + DATASET_PATH[:-1] + "Y" + str(epoch) + ".png"
            utils.saveimg(y_images, save_path)

        # Save Model Checkpoints
        if ((epoch % SAVE_MODEL_EVERY == 0) or (epoch == 1)):
            save_str = MODEL_SAVE_FOLDER + DATASET_PATH[:-1] + str(epoch)
            torch.save(Gxy.cpu().state_dict(), save_str + "_Gxy.pth")
            torch.save(Gyx.cpu().state_dict(), save_str + "_Gyx.pth")
            torch.save(Dxy.cpu().state_dict(), save_str + "_Dxy.pth")
            torch.save(Dxy.cpu().state_dict(), save_str + "_Dyx.pth")

            Gxy.to(device)
            Gyx.to(device)
            Dxy.to(device)
            Dyx.to(device)

        Gxy.train();
        Gyx.train();

    utils.plot_loss(loss_hist)

train()
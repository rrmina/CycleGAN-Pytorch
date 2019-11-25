import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
import random
import numpy as np

import models
import utils

from losses import real_loss, fake_loss, cycle_loss
from dataset import prepare_dataset

# Global Settings
TRAIN_IMAGE_SIZE = 128
DATASET_PATH = "summer2winter_yosemite/" # summer2winter_yosemite
X_CLASS = "A"
Y_CLASS = "B"
BATCH_SIZE = 4
SEED = 35

# Optimizer Settings
LR = 2e-4
BETA_1 = 0.5
BETA_2 = 0.999
NUM_EPOCHS = 1
CYCLE_WEIGHT = 10

def train():
    # Seeds
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare Dataset
    # prepare_dataset(DATASET_PATH)

    # Dataset and Dataloader
    transform = transforms.Compose([
        transforms.Resize(TRAIN_IMAGE_SIZE),
        transforms.CenterCrop(TRAIN_IMAGE_SIZE),
        transforms.ToTensor()
    ])
    x_trainloader, x_testloader = utils.prepare_loader(DATASET_PATH, X_CLASS, transform=transform, batch_size=BATCH_SIZE, shuffle=True)
    y_trainloader, y_testloader = utils.prepare_loader(DATASET_PATH, Y_CLASS, transform=transform, batch_size=BATCH_SIZE, shuffle=True)

    # Load Networks
    Gxy = models.Generator(64).to(device)
    Gyx = models.Generator(64).to(device)
    Dxy = models.Discriminator(64).to(device)
    Dyx = models.Discriminator(64).to(device)

    # Optimizer Settings
    G_param = list(Gxy.parameters()) + list(Gyx.parameters())
    G_optim = optim.Adam(G_param, lr=LR, betas=[BETA_1, BETA_2])
    Dxy_optim = optim.Adam(Dxy.parameters(), lr=LR, betas=[BETA_1, BETA_2])
    Dyx_optim = optim.Adam(Dyx.parameters(), lr=LR, betas=[BETA_1, BETA_2])

    # Fixed test samples
    x_testiter = iter(x_testloader)
    y_testiter = iter(y_testloader)
    fixed_X, _ = next(x_testiter)
    fixed_X = fixed_X.to(device)
    fixed_Y, _ = next(y_testiter)
    fixed_Y = fixed_Y.to(device)

    # Tensor to Image
    fixed_X_image = utils.ttoi(fixed_X)
    fixed_Y_image = utils.ttoi(fixed_Y)

    # Number of batches
    x_trainiter = iter(x_trainloader)
    y_trainiter = iter(y_trainloader)
    iter_per_epoch = min(len(x_trainiter), len(y_trainiter))

    # Output of generator is Tanh! so we need to scale real images accordingly
    def scale(tensor, mini=-1, maxi=1):
        return tensor * (maxi-mini) + mini

    # Outputs need to be scaled back from [mini, maxi] to [0, 1]
    def scale_back(tensor, mini=-1, maxi=1):
        return (tensor-mini)/(maxi-mini)

    # Training the CycleGAN
    for epoch in range(1, NUM_EPOCHS+1):
        print("========Epoch {}/{}========".format(epoch, NUM_EPOCHS))

        # Reset iterator otherwise we'll get unequal batch sizes
        x_trainiter = iter(x_trainloader)
        y_trainiter = iter(y_trainloader)

        for _ in range (iter_per_epoch-1):  # -1 in case of imbalanced sizes of the last batch

            # Fetch the dataset
            x_real, _ = next(x_trainiter)
            x_real = scale(x_real, -1, 1)
            x_real = x_real.to(device)
            y_real, _ = next(y_trainiter)
            y_real = scale(y_real, -1, 1)
            y_real = y_real.to(device)

            # ============= Generator ==============
            # Similar to training discriminator networks, in training 
            # generator networks, we fix discriminator networks.
            # However, cycle consistency prohibits us 
            # from training generators seperately.
     
            # Generator X -> Y Adversarial 
            G_optim.zero_grad()                         # Zero-out gradients
            Gxy_out = Gxy(x_real)                       # Gxy Forward Pass
            D_Gxy_out = Dxy(Gxy_out)                    # Gxy -> Dxy Forward
            Gxy_loss = real_loss(D_Gxy_out)             # Gxy Real Loss
            
            # Generator Y -> X Adversarial Loss
            Gyx_out = Gyx(y_real)                       # Gyx Forward Pass
            D_Gyx_out = Dyx(Gyx_out)                    # Gyx -> Dyx Forward
            Gyx_loss = real_loss(D_Gyx_out)             # Gyx Real Loss
            
            # Cycle Consistency Loss
            y_x_y = Gxy( Gyx(x_real) )                  # Reconstruct Y
            yxy_cycle_loss = cycle_loss(y_x_y, y_real)  # Y-X-Y Cycle Reconstruction Loss  
            x_y_x = Gyx( Gxy(y_real) )                  # Reconstruct X
            xyx_cycle_loss = cycle_loss(x_y_x, x_real)  # X-Y-X Cycle Reconstruction Loss

            # Generator Total Loss
            G_loss = Gxy_loss + Gyx_loss + CYCLE_WEIGHT * (xyx_cycle_loss + yxy_cycle_loss)
            G_loss.backward()
            G_optim.step() 

            # ========= Discriminator ==========
            # In training the discriminators, we fix the generators' parameters.
            # It is alright to train both discriminators seperately beceause
            # their forward pass don't share any parameters with each other

            # Discriminator X -> Y Adversarial Loss
            Dxy_optim.zero_grad()                       # Zero-out gradients
            Dxy_real_out = Dxy(y_real)                  # Dxy Forward Pass
            Dxy_real_loss = real_loss(Dxy_real_out)     # Dxy Eeal loss
            Dxy_fake_out = Dxy(Gxy(x_real))             # Gxy produces fake-y images
            Dxy_fake_loss = fake_loss(Dxy_fake_out)     # Dxy Fake Loss
            Dxy_loss = Dxy_real_loss + Dxy_fake_loss    # Dxy Total Loss
            Dxy_loss.backward()                         # Dxy Backprop
            Dxy_optim.step()                            # Dxy Gradient Descent

            # Discriminator Y-> X Adversarial Loss
            Dyx_optim.zero_grad()                       # Zero-out gradients
            Dyx_real_out = Dyx(x_real)                  # Dyx Forward Pass
            Dyx_real_loss = real_loss(Dyx_real_out)     # Dyx Eeal loss
            Dyx_fake_out = Dyx(Gyx(y_real))             # Gyx produces fake-x images
            Dyx_fake_loss = fake_loss(Dyx_fake_out)     # Dyx Fake Loss
            Dyx_loss = Dyx_real_loss + Dyx_fake_loss    # Dyx Total Loss
            Dyx_loss.backward()                         # Dyx Backprop
            Dyx_optim.step()                            # Dyx Gradient Descent

        # Print Losses
        print("Dxy Loss: {} Dyx Loss: {} Generator Loss: {}".format(Dxy_loss.item(), Dyx_loss.item(), G_loss.item()))
            
        # Generate Sample Fake Images
        Gxy.eval()
        Gyx.eval()
        with torch.no_grad():
            generated_y = Gyx(fixed_X)
            generated_y_img = utils.ttoi(generated_y.clone().detach())

            generated_x = Gxy(fixed_Y)
            generated_x_img = utils.ttoi(generated_x.clone().detach())
            
            H = W = TRAIN_IMAGE_SIZE
            concat_y = utils.concatenate_images(fixed_Y_image, generated_y_img, H, W)
            concat_x = utils.concatenate_images(fixed_X_image, generated_x_img, H, W)
    
            utils.saveimg(concat_x, "generated_x.png")
            utils.saveimg(concat_y, "generated_y.png")

train()
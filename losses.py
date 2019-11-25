import torch
import torch.nn as nn
import numpy as np

# For some reason, nn.MSELoss (and maybe nn.L1Loss too)
# throws error when using scalar values as target values
def real_loss(D_out):
    #criterion = nn.MSELoss()
    return torch.mean((D_out - 1)**2)
    
def fake_loss(D_out):
    #criterion = nn.MSELoss()
    return torch.mean(D_out**2)

def cycle_loss(generated, original):
    #criterion = nn.L1Loss()
    #return criterion(original, generated)
    return torch.mean(torch.abs(generated-original))
    
# Housekeeping Functions
def createLogger(names=[]):
    loss_hist = {}
    for name in names:
        loss_hist[name] = []

    return loss_hist

def updateEpochLogger(curr_loss_hist, values=[]):
    curr_loss_hist["Gxy"].append(values[0].item())
    curr_loss_hist["Gyx"].append(values[1].item())
    curr_loss_hist["Dxy"].append(values[2].item())
    curr_loss_hist["Dyx"].append(values[3].item())
    curr_loss_hist["cycle"].append(values[4].item())
    return curr_loss_hist

def updateGlobalLogger(loss_hist, curr_loss_hist):
    for key in loss_hist:
        loss_hist[key].append(np.mean(curr_loss_hist[key]))

    return loss_hist
import torch
import torch.nn as nn

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
    
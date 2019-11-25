import torch
import torch.nn as nn

def real_loss(x):
    #return torch.mean((x-1)**2)
    criterion = nn.MSELoss()
    return criterion(x, 1)
    

def fake_loss(x):
    #return torch.mean(x**2)
    criterion = nn.MSELoss()
    return criterion(x, 0)
    

def cycle_loss(out, target):
    #return torch.mean(torch.abs(out-target))
    criterion = nn.L1Loss()
    return criterion(out, target)
    
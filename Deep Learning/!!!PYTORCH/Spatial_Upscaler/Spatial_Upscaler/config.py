import numpy as np
import torch

def current_device(index:int=0):
    if torch.cuda.device_count() >= index + 1:
        return torch.device(f'cuda:{index}')
    return torch.device('cpu')
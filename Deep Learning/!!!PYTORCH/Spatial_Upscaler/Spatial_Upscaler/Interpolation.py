from typing import Optional, List, Tuple

import numpy as np

import torch
import torchvision


def nearest_neighbor_interpolation(tens:torch.tensor=torch.rand(2,2), new_size:Tuple[int, ...]=(8,8)) -> torch.tensor:
    '''
        Nearest neighbor interpolation

        Args:
            tens -> PyTorch tensor with arbitrary shape
            new_size -> Tuple with arbitrary shape

        Return:
            New PyTorch tensor with new values after nearest-neighbor interpolation
    '''
    assert tens.shape <= new_size, "Shape of new tensor '{0}' has to be equal or greator to original '{1}'".format(tens.shape, new_size)
    id_x = torch.arange(start=0, end=tens.shape[-1], step=(tens.shape[-1] / new_size[-1])).to(dtype=torch.int64)
    id_y = torch.arange(start=0, end=tens.shape[-2], step=(tens.shape[-2] / new_size[-2])).to(dtype=torch.int64)
    dest = torch.zeros(new_size).to(tens)
    dest[:] = tens[..., id_y,:][..., :, id_x]
    return dest

def linear_interpolation(tens:torch.tensor=torch.ones(2), new_size:Tuple[int, ...]=(4,)) -> torch.tensor:
    '''
        Linear Interpolation. Function interpolates only in 1D manner
        If tensor is 2D or more, then values will be interpolated in row manner
        
        Args:
            tens -> PyTorch tensor with arbitrary shape
            new_size -> Tuple with arbitrary shape

        Return:
            New PyTorch tensor with new values linearly interpolated
    '''
    assert tens.shape <= new_size, "Shape of new tensor '{0}' has to be equal or greator to original '{1}'".format(tens.shape, new_size)
    assert tens.dim() == len(new_size), "Tensor dim '{0}' has to be equal to new size dim '{1}'".format(tens.dim(), len(new_size))
    immediate_interp_values = torch.linspace(start=0., end=1., steps=new_size[-1]).unsqueeze(0)
    max_val = torch.max(tens[..., 0], tens[..., -1]).unsqueeze(1)
    min_val = torch.min(tens[..., 0], tens[..., -1]).unsqueeze(1)

    new_tens = torch.zeros(new_size).to(tens)
    new_tens[..., 0], new_tens[..., -1] = min_val[..., 0], max_val[..., -1]
    new_tens[:] = (max_val - min_val) * immediate_interp_values + min_val
    return new_tens



def bilinear_interpolation():
    pass
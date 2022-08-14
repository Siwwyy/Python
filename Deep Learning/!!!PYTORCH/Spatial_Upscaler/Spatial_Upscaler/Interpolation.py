from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision

from Utils import dim_indexing

def nearest_neighbor_interpolation(tens: torch.tensor=torch.rand(2, 2), new_size: Tuple[int, ...]=(8, 8)) -> torch.tensor:
    """
    Nearest neighbor interpolation
    Args:
        tens -> PyTorch tensor with arbitrary shape
        new_size -> Tuple with arbitrary shape
    Return:
        New PyTorch tensor with new values after nearest-neighbor interpolation
    """
    assert (tens.shape <= new_size), "Shape of new tensor '{0}' has to be equal or greator to original '{1}'".format(tens.shape, new_size)
    id_x = torch.arange(start=0, end=tens.shape[-1], step=(tens.shape[-1] / new_size[-1])).to(dtype=torch.int64)
    id_y = torch.arange(start=0, end=tens.shape[-2], step=(tens.shape[-2] / new_size[-2])).to(dtype=torch.int64)
    dest = torch.zeros(new_size).to(tens)
    dest[:] = tens[..., id_y, :][..., :, id_x]
    return dest


def linear_interpolation(tens: torch.tensor=torch.ones(2), new_size: Tuple[int, ...]=(4,), dim:int=-1) -> torch.tensor:
    """
    Linear Interpolation. Function interpolates only in 1D manner
    If tensor is 2D or more, then values will be interpolated in row manner

    Args:
        tens -> Tensor with arbitrary shape
        new_size -> Tuple with shape, dim as tens (shape at last pos should be greater, rest equal)
    Return:
        New Tensor with new values linearly interpolated
    """
    assert (tens.shape[-1] < new_size[-1]) & (tens.dim() == len(new_size)), "Shape of new tensor '{0}' has to be greater at last dim and equal at others to original '{1}'".format(tens.shape, new_size)
    assert dim <= tens.dim(), "Dim param '{0}' is greater than tensor dims '{1}'".format(dim, tens.dim())
    immediate_interp_values = torch.linspace(start=0.0, end=1.0, steps=new_size[-1]).unsqueeze(0)
    
    left_elem_idx = dim_indexing(tens.dim(), dim=dim, indice=0)
    right_elem_idx = dim_indexing(tens.dim(), dim=dim, indice=-1)
    left_val = tens[left_elem_idx].unsqueeze(1)
    right_val = tens[right_elem_idx].unsqueeze(1)

    new_tens = torch.zeros(new_size).to(tens)
    new_tens[left_elem_idx], new_tens[right_elem_idx] = left_val[left_elem_idx], right_val[right_elem_idx]
    new_tens[:] = (right_val - left_val) * immediate_interp_values + left_val
    return new_tens


def bilinear_interpolation(tens: torch.tensor=torch.ones(2,2), new_size: Tuple[int, ...]=(4,4)) -> torch.tensor:
    """
    Bilinear Interpolation. Function interpolates only in 2D manner
    Tensor will be interpolated in row direction and column

    Args:
        tens -> Tensor with arbitrary shape
        new_size -> Tuple with shape, dim as tens (shape at last pos should be greater, rest equal)
    Return:
        New Tensor with new values bilinearly interpolated
    """
    assert (tens.dim() == len(new_size)), "Shape of new tensor '{0}' has to be greater at last dim and equal at others to original '{1}'".format(tens.shape, new_size)




    new_tens = linear_interpolation(new_tens, new_size)

    new_tens = torch.zeros(new_size).to(tens)



    return new_tens

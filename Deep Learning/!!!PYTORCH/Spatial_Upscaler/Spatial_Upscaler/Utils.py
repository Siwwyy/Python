from typing import Union, List


import torch


def dim_indexing(tens_dim:int=2, dim:int=-1, indice:Union[int, slice]=-1) -> List[Union[int, slice]]: 
    """
    Function indexes through specified dim

    Args:
        tens_dim -> tensor dimensionality
        dim -> dimension to index through
        indice -> integer or slice to use on specified dimension
    Return:
        N-tens_dim slices list and at specified dim: Union[int, slice]
    """
    slices = [slice(None)] * tens_dim
    slices[dim] = indice
    return slices

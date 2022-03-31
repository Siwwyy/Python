from functools import partial
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import ssl
ssl._create_default_https_context = ssl._create_unverified_context



class CIFAR100_DS(torch.utils.data.Dataset):
    '''
        CIFAR 100 Dataset creator
    '''

    def __init__(self, root_dir:str="data", transforms:Optional[List]=None, 
                 train:bool=True, download:bool=False):
        super().__init__()
        self.train_ds = torchvision.datasets.CIFAR100(root=root_dir,
                                                        train=True,
                                                        transform=transforms,
                                                        download=download)
        self.test_ds = torchvision.datasets.CIFAR100(root=root_dir,
                                                        train=False,
                                                        transform=transforms,
                                                        download=download)


    def __getitem__(self, index:int=0):
        return 0

    def __len__(self) -> Tuple[torchvision.datasets.CIFAR100, torchvision.datasets.CIFAR100]:
        return (len(self.train_ds), len(self.test_ds))

    def get_ds(self) -> Tuple[torchvision.datasets.CIFAR100, torchvision.datasets.CIFAR100]:
        return (self.train_ds, self.test_ds)

    def get_train_ds(self) -> torchvision.datasets.CIFAR100:
        return self.train_ds

    def get_test_ds(self) -> torchvision.datasets.CIFAR100:
        return self.test_ds





    #def get_train_ds(self) -> torchvision.datasets.CIFAR100:
    #    if train_ds is None:
    #        raise ValueError("train dataset is set to false in constructor")
    #    return self.train_ds

    #def get_test_ds(self) -> torchvision.datasets.CIFAR100:
    #    if test_ds is None:
    #        raise ValueError("train dataset is set to true in constructor")
    #    return self.test_ds
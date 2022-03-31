'''
    Own includes
'''

from config import current_device
from Data_Set import CIFAR100_DS

########
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

from enum import Enum
from typing import Optional, List, Tuple


class EAvailable_Datasets(Enum):
    #CIFAR100 = CIFAR100_DS()
    CIFAR100 = 1


class Image_DataLoader(torch.utils.data.DataLoader):
    '''
        Image Dataloader
    '''
    def __init__(self, dataset, batch_size: Optional[int]=1,
                 shuffle: bool=False, sampler=None, batch_sampler=None,
                 num_workers=0, pin_memory=False):
        super().__init__(dataset, batch_size, shuffle, sampler, 
                         batch_sampler, num_workers, pin_memory)


    def __getitem__(self):
        pass




from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
    Load Dataset
'''
#world_countries_DS = pd.read_csv('data/World_Countries_DS.csv', delimiter=',', dtype=[np.str, np.float32, np.float32, np.str]) #load DS from csv file. I use comma as an delimeter
world_countries_DS = pd.read_csv('data/World_Countries_DS.csv', delimiter=',') #load DS from csv file. I use comma as an delimeter
print(world_countries_DS.values)


from pytorch_utils import ToTensor
to_tensor = ToTensor()
tens_world_countries_DS = to_tensor(world_countries_DS, columns=['latitude', 'longitude'], dtypes=['float', 'float'], device=device)

print(tens_world_countries_DS)
    

from operator import getitem
from model import Model_First
from typing import Dict, Tuple, List, Union, Optional
from pathlib import Path
from skimage import transform
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader

import torch
import numpy as np
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Custom_Dataset(torch.utils.data.Dataset):
    """Custom_Dataset"""

    def __init__(self, root_dir:str, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len([name for name in Path.iterdir(self.root_dir.absolute()) if Path.is_file(name)])

    def __getitem__(self, idx:Union[torch.tensor, int]=0) -> Dict[str, torch.tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        for file_id, file_name in enumerate(Path.iterdir(self.root_dir.absolute())):
            if file_id == idx:
                image = torch.from_numpy(imread(file_name)).permute(2,0,1).float()
                sample = {'image': image}
                if self.transform:
                    sample = self.transform(sample)
                return sample
        return None

'''
    Display sample images
'''
dataset = Custom_Dataset(root_dir="dataset/")

fig = plt.figure(figsize=(12,3))

for i in range(len(dataset)):
    sample = dataset[i]
    

    print(i, sample['image'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    #show_landmarks(**sample)
    plt.imshow(sample['image'].permute(1,2,0).int())

    if i == 3:
        plt.show()
        break


#########
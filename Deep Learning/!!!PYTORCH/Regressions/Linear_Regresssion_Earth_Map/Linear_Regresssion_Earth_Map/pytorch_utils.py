import os

import torch
import torch.nn as nn

import numpy as np
import pandas as pd


# determine the supported device
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU
    return device

class ToTensor(object):
    '''
        Converts numpy, pandas etc. to tensor

        Arguments:
            data: dataset or values to convert
            device: device of torch tensor (by default cpu)

        Returns:
            dictonary with torch tensor/s with specified dtype and device.'''
    def __call__(self, data, columns=[], dtypes=['float'], device:torch.device=torch.device("cpu")):


        #image, landmarks = sample['image'], sample['landmarks']

        ## swap color axis because
        ## numpy image: H x W x C
        ## torch image: C x H x W

        #return {'image': torch.from_numpy(image),
        #        'landmarks': torch.from_numpy(landmarks)}


        #if isinstance(data, pd.DataFrame):
        #    output_dic = dict()
        #    for idx, column in enumerate(data.columns):
        #        output_dic[column] =
        #        torch.from_numpy(data[column]).to(device=device)
        #    return output_dic
        #elif isinstance(data, np.ndarray):
        #    return torch.from_numpy(data).to(device=device)
        #else:
        #    return data
        output_dic = dict()
        if isinstance(data, pd.DataFrame):
            output_dic = dict()
            for idx, column in enumerate(columns):
                output_dic[column] = torch.from_numpy(data[column].values).to(device=device)

        return output_dic

        #landmarks = data.astype('float')
        #landmarks = np.array([landmarks])
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        #sample = {'image': image, 'landmarks': landmarks}

        #keys = list(data.columns)
        #output_dict = dict()
        #for idx, key in enumerate(keys):
        #        output_dic[column] = torch.from_numpy(data[key].values).to(device=device)
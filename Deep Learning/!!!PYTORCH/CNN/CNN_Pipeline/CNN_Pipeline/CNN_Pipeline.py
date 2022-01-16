import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from utils import get_CIFAR10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
    Prepare a dataset
'''
print('==> Preparing data..')
ds_train, ds_test = get_CIFAR10()
dls_train = torch.utils.data.DataLoader(ds_train, batch_size=128, shuffle=True, num_workers=0)
dls_test = torch.utils.data.DataLoader(ds_test, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

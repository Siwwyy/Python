
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


########################################################################################################


'''
    Load Dataset
'''
#world_countries_DS = pd.read_csv('data/World_Countries_DS.csv', delimiter=',', dtype=[np.str, np.float32, np.float32, np.str]) #load DS from csv file. I use comma as an delimeter
world_countries_DS = pd.read_csv('data/World_Countries_DS.csv', delimiter=',') #load DS from csv file. I use comma as an delimeter
#world_countries_DS = pd.read_csv('data/World_Cities_DS_large.csv', delimiter=',') #load DS from csv file. I use comma as an delimeter
#print(world_countries_DS.values)


from pytorch_utils import ToTensor
to_tensor = ToTensor()

data = to_tensor(world_countries_DS, columns=['latitude', 'longitude'], dtypes=['float', 'float'], device=device)
#print(data)

  
print(data['latitude'].shape)
train_data = torch.stack((data['longitude'], data['latitude']), dim=1)
print(train_data.shape)


########################################################################################################


'''
    Plotting
'''
def plot_xy(x:torch.tensor=torch.rand(100), y:torch.tensor=torch.rand(100)):
    plt.figure(figsize=(15,15))
    plt.scatter(x, y, color='r') 
    plt.show()

#plot_xy(train_data[:, 0].cpu(), train_data[:, 1].cpu())

def plot_world_countries(x:torch.tensor=torch.rand(100), y:torch.tensor=torch.rand(100), figsize=(20,10)):
    world_map = plt.imread('images/world_map.png')

    x_min = train_data[:, 0].min().item()
    x_max = train_data[:, 0].max().item()
    y_min = train_data[:, 1].min().item()
    y_max = train_data[:, 1].max().item()

    plt.figure(figsize=figsize)
    plt.plot(x, y, 'mo', markersize=2.0)
    #plt.imshow(world_map, extent=(-160, 160, -80, 80))
    #plt.imshow(world_map, extent=(x_min + 10., x_max + 8., y_min - 7., y_max + 3))
    #plt.imshow(world_map, extent=(x_min + 8., x_max + 10., y_min - 8., y_max + 2.))
    #plt.imshow(world_map, extent=(x_min + 0.08, x_max + 0.01, y_min - 0.08, y_max + 0.02))
    #plt.imshow(world_map, extent=(x_min + x_max * 0.07, x_max, y_min + y_min * 0.01, y_max))
    plt.imshow(world_map, extent=(x_min + 10, x_max, y_min, y_max))

    plt.grid(linestyle='dotted')
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    #plt.legend()
    plt.tight_layout()
    plt.show()


plot_world_countries(train_data[:, 0].cpu(), train_data[:, 1].cpu())


########################################################################################################


'''
    Dataset Normalization -> Z-norm
'''
def data_normalization(data:torch.tensor, mean_zero:bool=False) -> torch.tensor:
    mu, sigma = data.mean(), data.std()
    return (data - mu) / sigma

print(train_data.min(), train_data.max())
train_data = data_normalization(train_data)
print(train_data.min(), train_data.max())
#plot_world_countries(train_data[:, 0].cpu(), train_data[:, 1].cpu())
#plot_xy(train_data[:, 0].cpu(), train_data[:, 1].cpu())

########################################################################################################


'''
    Split train_data to train and test
'''
train_data_size = 240
x_train, y_train, x_test, y_test = train_data[:train_data_size, 0], train_data[:train_data_size, 1], train_data[train_data_size:, 0], train_data[train_data_size:, 1]


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


########################################################################################################


'''
    Init model
'''
# Linear regression
# f = a * x + b
a = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
b = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)


# model output
def forward(x, a, b):
    return a * x + b

# loss == MSE
def loss(y_hat, y):
    return ((y_hat - y) ** 2).mean()


########################################################################################################


'''
    Training Pipeline
'''
learning_rate = 1e-3
num_epochs = 1000

cost_per_epoch = []
for epoch in range(num_epochs):

    # predict == forward pass
    y_hat = forward(x_train, a, b)

    if x_train.is_leaf: #if x_train becomes a leaf (somehow), then just detach it from computional graph
        x_train.detach()

    #Calculate the loss
    epoch_loss = loss(y_hat, y_train)
    cost_per_epoch.append(epoch_loss.cpu().item())

    # calculate gradients == backward pass
    epoch_loss.sum().backward()

    # update weights
    #w.data = w.data - learning_rate * w.grad
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad

        # zero the gradients after updating
        a.grad.zero_()
        b.grad.zero_()

    #Log pass
    print('Epoch: %03d' % (epoch + 1), end="")
    print(' | Cost: %.3f' % epoch_loss)


if torch.cuda.is_available():
    a = a.to(device='cpu')
    b = b.to(device='cpu')

    x_train = x_train.to(device='cpu')
    y_train = y_train.to(device='cpu')
    x_test = x_test.to(device='cpu')
    y_test = y_test.to(device='cpu')

print(a)
print(b)

fig, axs = plt.subplots(figsize = (20,6))
plt.plot(cost_per_epoch)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


########################################################################################################


'''
    Inference Pipeline
'''

world_map = plt.imread('images/world_map.png')

x_min = train_data[:, 0].min().item()
x_max = train_data[:, 0].max().item()
y_min = train_data[:, 1].min().item()
y_max = train_data[:, 1].max().item()

plt.figure(figsize=(20,10))
plt.plot(x_train, y_train, 'mo', markersize=2.0)
plt.plot(x_train, forward(x_train, a, b).detach().numpy(), color='g')
plt.imshow(world_map, extent=(x_min + x_max * 0.07, x_max, y_min + y_min * 0.01, y_max))

plt.grid(linestyle='dotted')
plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])
plt.xticks([])
plt.yticks([])
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.tight_layout()
plt.show()


########################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import seaborn as sns

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


'''
  Data normalization
  https://www.baeldung.com/cs/normalizing-inputs-artificial-neural-network
'''

def data_normalization(data:torch.tensor) -> torch.tensor:
  mu, sigma = data.mean(), data.std()
  return (data - mu) / sigma



#Get dataset
from sklearn.datasets import load_breast_cancer

#data_cancer = load_breast_cancer()
#print(data_cancer.keys())

#x = data_cancer['data']
#y = data_cancer['target']
#names = data_cancer['target_names']
#feature_names = data_cancer['feature_names']

#x_data = torch.from_numpy(x).to(dtype=torch.float32, device=device)
#print("X shape: ", x_data.shape)

#y_data = torch.from_numpy(y).to(dtype=torch.int32, device=device)
#print("Y shape: ", y_data.shape)


#x_train, y_train = x_data[:, :2].clone(), y_data.clone()

#print(x_train.shape)
#print(y_train.shape)

def nonlinear_dataset(train_size:int=1000, test_size:int=100):

    #Train
    x_train = torch.unsqueeze(torch.linspace(-2, 2, train_size), dim=1)  
    y_train = x_train.pow(2) + 0.2*torch.rand(x_train.size()) 

    #Test
    x_test = torch.unsqueeze(torch.linspace(-2, 2, test_size), dim=1)  
    y_test = x_test.pow(2) + 0.2*torch.rand(x_test.size()) 

    return (x_train, y_train, x_test, y_test)


x_train, y_train, x_test, y_test = nonlinear_dataset(1000, 1000)


def plot_xy(x:torch.tensor=torch.rand(100), y:torch.tensor=torch.rand(100)):
    plt.figure(figsize=(15,15))
    plt.scatter(x, y, color='b') 
    plt.show()

#plot_xy(x_train, y_train)

#plot_xy(x_train_normalized.cpu())

'''
    Autograd Gradients
'''

class Logistic_Regression(nn.Module):
    def __init__(self, num_features:int=1, num_classes:int=1):
        super().__init__()
        self.hidden = nn.Linear(num_features, 10, device=device)
        self.activation = nn.ReLU()
        self.out = nn.Linear(10, num_classes, device=device)

    def forward(self, x):
        output = self.hidden(x)
        output = self.activation(output)
        output = self.out(output)
        return output

    def evaluate(self, x, y):
        y_hat = self.forward(x)
        accuracy = (torch.isclose(y_hat, y, atol=0.2).sum() / y.size(0)) * 100.
        return accuracy

    def train_pass(self, 
                   x:torch.tensor, 
                   y:torch.tensor, 
                   num_epochs:torch.int32=10,
                   lr:torch.float32=0.001):

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        cost_per_epoch = []
        for epoch in range(num_epochs):

            #Zero gradients
            optimizer.zero_grad()

            #Forward pass
            pred = self.forward(x)

            #Calculate the loss
            loss = criterion(pred, y)
            cost_per_epoch.append(loss.cpu().detach().numpy())

            #Compute gradients
            loss.backward()
            optimizer.step()

            #Log pass
            print('Epoch: %03d' % (epoch + 1), end="")
            print(' | Train ACC: %.2f' % self.evaluate(x, y), end="%")
            print(' | Cost: %.3f' % loss)

        return cost_per_epoch

    def train_pass_sgd(self, 
                       x:torch.tensor, 
                       y:torch.tensor, 
                       num_epochs:torch.int32=10,
                       lr:torch.float32=0.001):

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        cost_per_epoch = []
        for epoch in range(num_epochs):
            loss_per_elem = 0.
            for x_i, y_i in zip(x, y):

                #Zero gradients
                optimizer.zero_grad()

                #Forward pass
                pred_i = self.forward(x_i)

                #Calculate the loss
                loss = criterion(pred_i, y_i)
                cost_per_epoch.append(loss.cpu().detach().numpy())

                loss_per_elem = max(loss_per_elem, loss)

                #Compute gradients
                loss.backward()
                optimizer.step()

            #Log pass
            print('Epoch: %03d' % (epoch + 1), end="")
            print(' | Train ACC: %.2f' % self.evaluate(x, y), end="%")
            print(' | Cost: %.3f' % loss_per_elem)

        return cost_per_epoch

    def train_pass_mb(self, 
                       x:torch.tensor, 
                       y:torch.tensor, 
                       num_epochs:torch.int32=10,
                       lr:torch.float32=0.001):

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        batch_size = 100
        n_batches = x.size(0) // batch_size

        cost_per_epoch = []
        for epoch in range(num_epochs):
            loss_per_elem = 0.
            for batch in range(n_batches):
                x_i, y_i = x[batch*batch_size:(batch+1)*batch_size], y[batch*batch_size:(batch+1)*batch_size]

                #Zero gradients
                optimizer.zero_grad()

                #Forward pass
                pred_i = self.forward(x_i)

                #Calculate the loss
                loss = criterion(pred_i, y_i)
                cost_per_epoch.append(loss.cpu().detach().numpy())

                loss_per_elem = max(loss_per_elem, loss)

                #Compute gradients
                loss.backward()
                optimizer.step()

            #Log pass
            print('Epoch: %03d' % (epoch + 1), end="")
            print(' | Train ACC: %.2f' % self.evaluate(x_i, y_i), end="%")
            print(' | Cost: %.3f' % loss_per_elem)

        return cost_per_epoch

x_train = x_train.to(device=device)
y_train = y_train.to(device=device)

net = Logistic_Regression(x_train.size(1))


if torch.cuda.is_available():
    x_train = x_train.to(device='cpu')
    y_train = y_train.to(device='cpu')

    net = net.to(device='cpu')


#fig = plt.figure(figsize=(15,15))
## Plotting both the curves simultaneously
#plt.scatter(x_train, y_train, color='b')
#plt.plot(x_train, net.forward(x_train).detach().numpy(), color='r', linewidth=5)
## To load the display window
#plt.show()


from functools import partial

##Hiperparams
num_epochs = 500
lr = 0.01

optimizers_list = [
    partial(net.train_pass, x=x_train, y=y_train, num_epochs=num_epochs, lr=lr),
    partial(net.train_pass_sgd, x=x_train, y=y_train, num_epochs=num_epochs, lr=0.001),
    partial(net.train_pass_mb, x=x_train, y=y_train, num_epochs=num_epochs, lr=0.001)
    ]

for optimizer in optimizers_list:
    print("=========================")
    loss = optimizer()

    # summarize history for loss
    fig, axs = plt.subplots(figsize = (20,6))
    plt.plot(loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    fig = plt.figure(figsize=(15,15))
    # Plotting both the curves simultaneously
    plt.scatter(x_train, y_train, color='b')
    plt.plot(x_train, net.forward(x_train).detach().numpy(), color='r', linewidth=5)
    # To load the display window
    plt.show()
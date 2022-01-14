import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def linear_dataset(train_size:int=1000, test_size:int=100):

    #Train
    a = 0.22
    b = 0.78
    x_train = torch.normal(0.0, 0.5, size=(train_size, 1))
    y_train = (a * x_train + b) + torch.normal(0.0, 0.1, size=(train_size, 1))

    #Test
    a = 0.1
    b = 0.4
    x_test = torch.normal(0.0, 0.5, size=(test_size, 1))
    y_test = (a * x_test + b) + torch.normal(0.0, 0.1, size=(test_size, 1))

    return (x_train, y_train, x_test, y_test)



x_train, y_train, x_test, y_test = linear_dataset(1000, 1000)


def plot_xy(x:torch.tensor=torch.rand(100), y:torch.tensor=torch.rand(100)):
    plt.figure(figsize=(15,15))
    plt.scatter(x, y, color='r') 
    plt.show()

plot_xy(x_train, y_train)


def data_normalization(data:torch.tensor, mean_zero:bool=False) -> torch.tensor:
    mu, sigma = data.mean(), data.std()
    return (data - 0.) / sigma if mean_zero else (data - mu) / sigma


# Here we replace the manually computed gradient with autograd
# Linear regression
# f = a * x + b
a = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
b = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)


# model output
def forward(x, a, b):
    return a * x + b

# loss = MSE
def loss(y, y_hat):
    return ((y_hat - y) ** 2).mean()

# Training
learning_rate = 0.01
n_iters = 1000

#Normalize the data
#x_train = data_normalization(x_train)
#y_train = data_normalization(y_train)

#plot_xy(x_train, y_train)

x_train = x_train.to(device=device)
y_train = y_train.to(device=device)

for epoch in range(n_iters):
    # predict = forward pass
    y_hat = forward(x_train, a, b)

    if x_train.is_leaf: #if x_train becomes a leaf (somehow), then just detach it from computional graph
        x_train.detach()

    # loss
    l = loss(y_hat, y_train)

    # calculate gradients = backward pass
    l.sum().backward()
    
    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: loss = {l.item():.8f}')

    # update weights
    #w.data = w.data - learning_rate * w.grad
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad

        # zero the gradients after updating
        a.grad.zero_()
        b.grad.zero_()


if torch.cuda.is_available():
    a = a.to(device='cpu')
    b = b.to(device='cpu')

    x_train = x_train.to(device='cpu')
    y_train = y_train.to(device='cpu')


print(a)
print(b)

fig = plt.figure(figsize=(15,15))
# Plotting both the curves simultaneously
plt.scatter(x_train, y_train, color='r')
plt.plot(x_train, forward(x_train, a, b).detach().numpy(), color='g')
# To load the display window
plt.show()


## Create tensors.
#x = torch.tensor(3.)
#w = torch.tensor(4., requires_grad=True)
#b = torch.tensor(5., requires_grad=True)


## Print tensors
#print(x)
#print(w)
#print(b)


## Arithmetic operations
#y = w * x + b
#print(y)

## Compute gradients
#y.backward()

## Display gradients
#print('dy/dw:', w.grad)
#print('dy/db:', b.grad)
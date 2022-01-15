import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets
from torchvision.transforms import ToTensor


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#Get Dataset
def load_data():
    transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                ])

    train_data = datasets.MNIST('data',
                                train=True,
                                download=False,
                                transform=transform)
    test_data = datasets.MNIST('data',
                                train=False,
                                download=False,
                                transform=transform)

    return train_data, test_data

#Plot MNIST dataset
def plot_MNIST(data:torch.tensor):
    figure = plt.figure(figsize=(10, 8))
    cols, rows = 5, 5
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img, label = data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


import time

def train_pass(model:nn.Module, data_loader:dict, num_epochs:int=1, lr:float=0.001):
    # Train the model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)   
    cost_per_epoch = []
    valid_per_epoch = []
        
    total_step = len(data_loader['train'])

    #Measure execution time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    overall_time = 0.0
    for epoch in range(num_epochs):
        train_loss = 0.0
        start_time = time.perf_counter()
        for i, (images, labels) in enumerate(data_loader['train']):

            # clear gradients for this training step   
            optimizer.zero_grad()    

            # gives batch data, normalize x when iterate train_loader
            b_x = images.to(device=device)   # batch x
            b_y = labels.to(device=device)   # batch y

            output = model.forward(b_x)             
            loss = criterion(output, b_y)
            current_loss = loss.item() * b_x.size(0)
            train_loss += current_loss

            # backpropagation, compute gradients 
            loss.backward()    

            # apply gradients             
            optimizer.step() 

            if (i+1) % 100 == 0:
                print('Epoch [{0}/{1}], Step [{2}/{3}], Train Loss: {4:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, current_loss))

        valid_loss = 0.0 
        for i, (images, labels) in enumerate(data_loader['valid']):
            # gives batch data, normalize x when iterate train_loader
            b_x = images.to(device=device)   # batch x
            b_y = labels.to(device=device)   # batch y
         
            # Forward Pass
            output = model.forward(b_x)             
            loss = criterion(output, b_y)

            # Calculate Loss
            current_loss = loss.item() * b_x.size(0)
            valid_loss += current_loss

        #cost_per_epoch.append(train_loss)   
        #valid_per_epoch.append(valid_loss)

        cost_per_epoch.append(train_loss / total_step)   
        valid_per_epoch.append(valid_loss / total_step)

        #Time performance measurement
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        shuffle_time = end_time - start_time
        overall_time += shuffle_time
        print('Epoch [{0}/{1}], Train Loss: {2:.4f}, Valid Loss: {3:.4f}, Elapsed time: {4:.1f} seconds'.format(epoch + 1, num_epochs, cost_per_epoch[-1], valid_per_epoch[-1], shuffle_time))
        print()

    print('Overall time: {:.1f} seconds'.format(overall_time))

    return cost_per_epoch, valid_per_epoch


from matplotlib.ticker import MaxNLocator

# summarize history for loss
def plot_loss_valid(train_loss:list=[], valid_loss:list=[], epochs:int=10):
    fig, ax = plt.subplots(figsize = (20,6))
    plt.plot(range(1,epochs+1), train_loss, 'b', label='Training loss')
    plt.plot(range(1,epochs+1), valid_loss, 'c', label='Validation loss')
    plt.title('Training and validation accuracy')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()


def test_pass(model:nn.Module, data_loader:dict):
    # Test the model
    num_correct = torch.zeros(1, dtype=torch.float32, device=device)
    total = 0
    total_accuracy = []

    for i, (images, labels) in enumerate(data_loader['test']):
        # gives batch data, normalize x when iterate train_loader       
        b_x = images.to(device=device)   # batch x, inputs
        b_y = labels.to(device=device)   # batch y, targets
        pred, conv_out = model.predict(b_x)

        num_correct += torch.eq(pred, b_y).sum()
        total += labels.size(0)

    total_accuracy = torch.div(num_correct, float(total)) * 100.

    print("Test Accuracy of the model: {:.2f}%".format(total_accuracy.item()))

    return total_accuracy
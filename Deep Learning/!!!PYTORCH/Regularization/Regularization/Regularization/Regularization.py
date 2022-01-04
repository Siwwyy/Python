from functools import partial

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
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

#Create split to train and test data
train_data, test_data = load_data()

print(train_data)
print(test_data)

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


#plot_MNIST(train_data)


from torch.utils.data import DataLoader

train_data_split, valid_data = random_split(train_data,[50000,10000])

data_loaders_dict = {
    'train' : torch.utils.data.DataLoader(train_data_split, 
                                          batch_size=100, 
                                          shuffle=True),

    'valid' : torch.utils.data.DataLoader(valid_data, 
                                          batch_size=100, 
                                          shuffle=True),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True)
}

class Softmax_Regression(nn.Module):
    def __init__(self, num_classes:int=3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1,              
                                out_channels=16,            
                                kernel_size=5,              
                                stride=1,                   
                                padding=2, 
                                device=device)
        self.activation_conv1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=16,              
                                out_channels=32,            
                                kernel_size=5,              
                                stride=1,                   
                                padding=2,
                                device=device)
        self.activation_conv2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # fully connected layer, output 10 classes
        self.out = nn.Linear(in_features=(32 * 7 * 7), 
                             out_features=num_classes, 
                             device=device)

    def forward(self, x) -> torch.tensor:
        output = self.conv1(x)
        output = self.activation_conv1(output)
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.activation_conv2(output)
        output = self.pool2(output)

        #Fully connected layer
        #flatten the array, (N,W)
        output = output.view(output.size(0), -1)
        output = self.out(output)
        return output


class Softmax_Regression_Dropout(nn.Module):
    def __init__(self, num_classes:int=3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1,              
                                out_channels=16,            
                                kernel_size=5,              
                                stride=1,                   
                                padding=2, 
                                device=device)
        self.activation_conv1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=16,              
                                out_channels=32,            
                                kernel_size=5,              
                                stride=1,                   
                                padding=2,
                                device=device)
        self.activation_conv2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(p=0.2)
        # fully connected layer, output 10 classes
        self.out = nn.Linear(in_features=(32 * 7 * 7), 
                             out_features=num_classes, 
                             device=device)

    def forward(self, x) -> torch.tensor:
        output = self.conv1(x)
        output = self.activation_conv1(output)
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.activation_conv2(output)
        output = self.pool2(output)

        #Fully connected layer
        #flatten the array, (N,W)
        output = output.view(output.size(0), -1)

        #Apply dropout
        output = self.dropout(output)

        output = self.out(output)
        return output




n_classes = len(train_data.classes)

net = Softmax_Regression(n_classes)
net = net.to(device=device)

net_drop = Softmax_Regression_Dropout(n_classes)
net_drop = net_drop.to(device=device)

def train_pass(model:nn.Module, data_loader:dict, num_epochs:int=1, lr:float=0.001, regularization:str='none'):
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)   
    cost_per_epoch = []
    valid_per_epoch = []
        
    # Train the model
    total_step = len(data_loader['train'])
    
    if regularization == 'none':
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (images, labels) in enumerate(data_loader['train']):

                # clear gradients for this training step   
                optimizer.zero_grad()    

                # gives batch data, normalize x when iterate train_loader
                b_x = images.to(device=device)   # batch x
                b_y = labels.to(device=device)   # batch y

                output = model.forward(b_x)             
                loss = criterion(output, b_y)

                running_loss =+ loss.item() * b_x.size(0)

                # backpropagation, compute gradients 
                loss.backward()    

                # apply gradients             
                optimizer.step()   

                if (i+1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, running_loss))
            
            valid_loss = 0.0 
            for i, (images, labels) in enumerate(data_loader['valid']):
                # gives batch data, normalize x when iterate train_loader
                b_x = images.to(device=device)   # batch x
                b_y = labels.to(device=device)   # batch y
         
                # Forward Pass
                output = model.forward(b_x)             
                loss = criterion(output, b_y)

                # Calculate Loss
                valid_loss += loss.item() * b_x.size(0)

            cost_per_epoch.append(running_loss)   
            valid_per_epoch.append(valid_loss)

    elif regularization == 'L1':
        lambda_L1 = 0.001
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (images, labels) in enumerate(data_loader['train']):

                # clear gradients for this training step   
                optimizer.zero_grad()    

                # gives batch data, normalize x when iterate train_loader
                b_x = images.to(device=device)   # batch x
                b_y = labels.to(device=device)   # batch y

                output = model.forward(b_x)             
                loss = criterion(output, b_y)

                #L1 Loss
                regularization_loss = 0.0
                for param in model.parameters():
                    regularization_loss += torch.sum(torch.abs(param))
                loss = loss + regularization_loss*lambda_L1

                running_loss =+ loss.item() * b_x.size(0)



                # backpropagation, compute gradients 
                loss.backward()    

                # apply gradients             
                optimizer.step()   

                if (i+1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, running_loss))
            
            valid_loss = 0.0 
            for i, (images, labels) in enumerate(data_loader['valid']):
                # gives batch data, normalize x when iterate train_loader
                b_x = images.to(device=device)   # batch x
                b_y = labels.to(device=device)   # batch y
         
                # Forward Pass
                output = model.forward(b_x)             
                loss = criterion(output, b_y)

                # Calculate Loss
                valid_loss += loss.item() * b_x.size(0)

            cost_per_epoch.append(running_loss)   
            valid_per_epoch.append(valid_loss)

    elif regularization == 'L2':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.4)  #L2 Regularization
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (images, labels) in enumerate(data_loader['train']):

                # clear gradients for this training step   
                optimizer.zero_grad()    

                # gives batch data, normalize x when iterate train_loader
                b_x = images.to(device=device)   # batch x
                b_y = labels.to(device=device)   # batch y

                output = model.forward(b_x)             
                loss = criterion(output, b_y)

                running_loss =+ loss.item() * b_x.size(0)

                # backpropagation, compute gradients 
                loss.backward()    

                # apply gradients             
                optimizer.step()   

                if (i+1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, running_loss))
            
            valid_loss = 0.0 
            for i, (images, labels) in enumerate(data_loader['valid']):
                # gives batch data, normalize x when iterate train_loader
                b_x = images.to(device=device)   # batch x
                b_y = labels.to(device=device)   # batch y
         
                # Forward Pass
                output = model.forward(b_x)             
                loss = criterion(output, b_y)

                # Calculate Loss
                valid_loss += loss.item() * b_x.size(0)

            cost_per_epoch.append(running_loss)   
            valid_per_epoch.append(valid_loss)


    return cost_per_epoch, valid_per_epoch

epochs = 1
#loss, valid_loss = train_pass(net, data_loaders_dict, num_epochs=epochs, lr=0.001)



# summarize history for loss
def plot_loss_valid(loss, valid_loss, epochs):
  fig, axs = plt.subplots(figsize = (20,6))
  plt.plot(range(1,epochs+1), loss, 'b', label='Training loss')
  plt.plot(range(1,epochs+1), valid_loss, 'b', label='Validation loss')
  plt.title('Training and validation accuracy')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'valid'], loc='upper left')
  plt.show()



for model in [net, net_drop]:
    for regularizer in ['none', 'L1', 'L2']:
        print("=============================")
        print(type(model).__name__)
        print("Regularizer: {0}".format(regularizer))
        print("=============================")
        loss, valid_loss = train_pass(model, data_loaders_dict, num_epochs=epochs, lr=0.001, regularization=regularizer)
        plot_loss_valid(loss, valid_loss, epochs)
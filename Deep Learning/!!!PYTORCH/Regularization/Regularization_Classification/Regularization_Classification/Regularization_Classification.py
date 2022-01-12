
from Functions import load_data, plot_MNIST, train_pass, plot_loss_valid, test_pass

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Create split to train and test data
train_data, test_data = load_data()

print(train_data)
print(test_data)

#plot_MNIST(train_data)


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

class MNIST_Classifier(nn.Module):
    def __init__(self, input_shape:torch.Size=(1,1,28,28), num_classes:int=3):
        super().__init__()
        #torch.Size([100, 1, 28, 28])
        assert len(input_shape) >= 3, "Invalid shape {0}, should be CHW or NCHW".format(input_shape)

        self.input_shape = (input_shape[-1] * input_shape[-2] * input_shape[-3])

        self.l_hidden1 = nn.Linear(in_features=self.input_shape, out_features=self.input_shape)
        self.act_hidden1 = nn.ReLU()

        #self.l_hidden2 = nn.Linear(in_features=self.l_hidden1.out_features, out_features=self.input_shape)
        #self.act_hidden2 = nn.ReLU()

        #self.l_hidden3 = nn.Linear(in_features=self.l_hidden2.out_features, out_features=input_shape[-1])
        #self.act_hidden3 = nn.ReLU()

        self.l_output = nn.Linear(in_features=self.l_hidden1.out_features, out_features=num_classes)
        self.act_output = nn.ReLU()

    def forward(self, x) -> torch.tensor:
        x = x.view(-1, self.input_shape) #Flatten the input, to have N,F -> N: number of samples in batch, F: amount of input features 

        output = self.l_hidden1(x)
        output = self.act_hidden1(output)

        #output = self.l_hidden2(output)
        #output = self.act_hidden2(output)

        #output = self.l_hidden3(output)
        #output = self.act_hidden3(output)

        output = self.l_output(output)
        #output = self.act_output(output)
        return output

    def predict(self, x):
        probas = self.forward(x)
        labels = torch.argmax(probas, dim=1)
        return labels


        
sample = next(iter(data_loaders_dict['train']))
image, _ = sample
n_classes = len(train_data.classes)
net = MNIST_Classifier(image.shape, n_classes)
net = net.to(device=device)

epochs = 50

#loss, valid_loss = train_pass(net, data_loaders_dict, num_epochs=epochs, lr=0.0001)
#plot_loss_valid(loss, valid_loss, epochs)

#test_pass(net, data_loaders_dict)



from training_pipeline import *


loss, valid_loss = main_pipeline(net, data_loaders_dict, num_epochs=epochs, lr=0.0001)
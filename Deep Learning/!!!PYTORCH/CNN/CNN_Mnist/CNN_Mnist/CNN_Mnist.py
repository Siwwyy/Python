
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

        #conv1
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=16,            
                               kernel_size=5,              
                               stride=1,                   
                               padding=2).to(dtype=torch.float16)
        self.conv1_act = nn.ReLU()
        self.conv1_max_pool = nn.MaxPool2d(kernel_size=2)

        #conv2
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=32,            
                               kernel_size=5,              
                               stride=1,                   
                               padding=2)
        self.conv2_act = nn.ReLU()
        self.conv2_max_pool = nn.MaxPool2d(kernel_size=2)

        #output layer
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x) -> torch.tensor:

        #conv1
        output = self.conv1(x)
        output = self.conv1_act(output)
        output = self.conv1_max_pool(output)

        #conv2
        output = self.conv2(output)
        output = self.conv2_act(output)
        output = self.conv2_max_pool(output)

        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = output.view(x.size(0), -1)       
        y_hat = self.out(output)
        return y_hat, output    # return output for visualization

    def predict(self, x):
        probas, conv_output = self.forward(x)
        labels = torch.argmax(probas, dim=1)
        return labels


        
sample = next(iter(data_loaders_dict['train']))
image, _ = sample
n_classes = len(train_data.classes)
net = MNIST_Classifier(image.shape, n_classes)
print(net)
net = net.to(device=device)

epochs = 5

#loss, valid_loss = train_pass(net, data_loaders_dict, num_epochs=epochs, lr=0.0001)
#plot_loss_valid(loss, valid_loss, epochs)

#test_pass(net, data_loaders_dict)



from training_pipeline import *


loss, valid_loss = main_pipeline(net, data_loaders_dict, num_epochs=epochs, lr=0.0001)
plot_loss_valid(loss, valid_loss, epochs)


#Inference
import matplotlib.pyplot as plt


acc = inference(net, data_loaders_dict)

num_row = 5
num_col = 5# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(2*num_col,2*num_row))
fig.suptitle('Model accuracy {:.2f}%'.format(acc), fontsize=14, fontweight='bold')
for i in range(num_row):
    for j in range(num_col):
        ax = axes[i, j]
        img, label = test_data[i * 5 + j]
        ax.imshow(img.squeeze(0), cmap='gray')
        ax.set_title('Pred: {0} | True: {1}'.format(net.predict(img.unsqueeze(0).to(device=device)).item(), label))
        ax.set_xticks([])
        ax.set_yticks([])
plt.tight_layout()
plt.show()
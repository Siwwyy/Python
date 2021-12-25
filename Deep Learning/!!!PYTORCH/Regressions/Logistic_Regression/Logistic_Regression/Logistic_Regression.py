#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import matplotlib.pyplot as plt
#import seaborn as sns

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#from sklearn.datasets import load_breast_cancer

##Get dataset
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

##print(names)
##print(feature_names)
##print(data_cancer.DESCR)

##print(y_data)
#'''
#  Data normalization
#  https://www.baeldung.com/cs/normalizing-inputs-artificial-neural-network
#'''

#def data_normalization(data:torch.tensor) -> torch.tensor:
#  mu, sigma = data.mean(), data.std()
#  return (data - mu) / sigma

#'''
#    Manual Gradients
#'''
#def BCE_loss(pred, target):
#    return -(torch.log(pred) * target + (1. - target) * torch.log(1. - pred)).mean()


#amount_of_samples = 560
#x_data_normalized = data_normalization(x_data)[:amount_of_samples, :2]
#y_data_normalized = y_data[:amount_of_samples]

##class Logistic_Regression(nn.Module):
##    def __init__(self, num_features:torch.int32=1):
##        super(Logistic_Regression, self).__init__()
##        self.num_features = num_features
##        self.weights = torch.rand((1, num_features), dtype=torch.float32, device=device)
##        self.bias = torch.ones(1, dtype=torch.float32, device=device)

##    def sigmoid(self, x):
##        return 1. / (1. + torch.exp(-x))

##    def forward(self, x):
##        output = torch.sum(torch.mul(x, self.weights), dim=1) + self.bias
##        output = self.sigmoid(output)
##        return output

##    def loss(self, pred, target):
##        return BCE_loss(pred, target)

##    def backward(self, x, pred, target):
##        pred_minus_target = torch.subtract(pred, target)
##        dw = torch.sum(torch.mul(pred_minus_target.unsqueeze(1), x))
##        db = torch.sum(pred_minus_target)
##        return dw, db

##    def predict(self, x):
##        pred = self.forward(x)
##        labels = torch.where(pred < 0.5, 0, 1)
##        return labels

##    def evaluate(self, x, y):
##        labels = self.predict(x)
##        if y.dim() == 0:
##            y = y.unsqueeze(0)
##        accuracy = (torch.sum(torch.eq(labels, y)).float() / y.size(0)) * 100.
##        return accuracy


##    def train_pass(self, x, y, 
##                   num_epochs:torch.int32=10,
##                   lr:torch.float32=0.001):

##        cost_per_epoch = []
##        for epoch in range(num_epochs):

##            #Forward pass
##            pred = self.forward(x)

##            #Calculate the loss
##            loss = self.loss(pred, y)
##            cost_per_epoch.append(loss.cpu())

##            #Compute gradients
##            grad_w, grad_b = self.backward(x, pred, y)

##            #Update weights
##            self.weights = self.weights - lr * grad_w
##            self.bias = self.bias - lr * grad_b

##            #Log pass
##            print('Epoch: %03d' % (epoch + 1), end="")
##            print(' | Train ACC: %.2f' % self.evaluate(x, y), end="%")
##            print(' | Cost: %.3f' % loss)

##        return cost_per_epoch

##    def train_pass_sgd(self, x, y, 
##                   num_epochs:torch.int32=10,
##                   lr:torch.float32=0.001):

##        cost_per_epoch = []
##        for epoch in range(num_epochs):
##            #loss = 0
##            change_w = 0.0
##            change_b = 0.0
##            for x_i, y_i in zip(x, y):
##                #Forward pass
##                pred = self.forward(x_i)

##                #Calculate the loss
##                loss = self.loss(pred, y_i)
##                cost_per_epoch.append(loss.cpu())

##                #Compute gradients
##                grad_w, grad_b = self.backward(x_i, pred, y_i)

##                ##Update weights
##                self.weights = self.weights - lr * grad_w
##                self.bias = self.bias - lr * grad_b

##                #Log pass
##                print('Epoch: %03d' % (epoch + 1), end="")
##                print(' | Train ACC: %.2f' % self.evaluate(x, y), end="%")
##                print(' | Cost: %.3f' % loss)

##        return cost_per_epoch




##net = Logistic_Regression(x_data_normalized.size(1))
###loss = net.train_pass(x_data_normalized, y_data_normalized, 300, 0.05)
##loss = net.train_pass_sgd(x_data_normalized, y_data_normalized, 300, 0.05)


### summarize history for loss
##fig, axs = plt.subplots(figsize = (20,6))
##plt.plot(loss)
##plt.title('model loss')
##plt.ylabel('loss')
##plt.xlabel('epoch')
##plt.legend(['train', 'test'], loc='upper left')
##plt.show()




#'''
#    Autograd Gradients
#'''

#class Logistic_Regression2(nn.Module):
#    def __init__(self, num_features:torch.int32=1):
#        super().__init__()
#        self.linear = nn.Linear(num_features, 1, device=device)

#    def forward(self, x):
#        output = torch.sigmoid(self.linear(x))
#        return output

#    def predict(self, x):
#        pred = self.forward(x)
#        labels = torch.where(pred < 0.5, 0, 1)
#        return labels

#    def evaluate(self, x, y):
#        labels = self.predict(x)
#        if y.dim() == 0:
#            y = y.unsqueeze(0)
#        accuracy = torch.sum(torch.eq(labels.squeeze(1), y)).float() / y.size(0)
#        return accuracy

#    def train_pass(self, x, y, 
#                   num_epochs:torch.int32=10,
#                   lr:torch.float32=0.001):

#        criterion = nn.BCELoss()
#        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

#        cost_per_epoch = []
#        for epoch in range(num_epochs):

#            #Forward pass
#            pred = self.forward(x)

#            #Calculate the loss
#            loss = criterion(pred.squeeze(1), y.float())
#            cost_per_epoch.append(loss.cpu().detach().numpy())

#            #Compute gradients
#            loss.backward()
#            optimizer.step()

#            #Zero gradients
#            optimizer.zero_grad()
                
#            #Log pass
#            print('Epoch: %03d' % (epoch + 1), end="")
#            print(' | Train ACC: %.2f' % self.evaluate(x, y), end="%")
#            print(' | Cost: %.3f' % loss)

#        return cost_per_epoch

#    def train_pass_sgd(self, x, y, 
#                num_epochs:torch.int32=10,
#                lr:torch.float32=0.001):

#        criterion = nn.BCELoss()
#        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.02)

#        cost_per_epoch = []
#        for epoch in range(num_epochs):
#            for x_i, y_i in zip(x, y):
#                #Forward pass
#                pred_i = self.forward(x_i)

#                #Calculate the loss
#                loss = criterion(pred_i.squeeze(0), y_i.float())
#                cost_per_epoch.append(loss.cpu().detach().numpy())
#                #Zero gradients
#                optimizer.zero_grad()

#                #Compute gradients
#                loss.backward()
#                optimizer.step()

#                #Log pass
#                #print('Epoch: %03d' % (epoch + 1), end="")
#                #print(' | Train ACC: %.2f' % self.evaluate(x_i, y_i), end="%")
#                #print(' | Cost: %.3f' % loss)

#        return cost_per_epoch

#    def train_pass_mb(self, x, y, 
#                num_epochs:torch.int32=10,
#                lr:torch.float32=0.001):

#        criterion = nn.BCELoss()
#        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

#        batch_size = 100
#        n_batches = x.size(0) // batch_size

#        cost_per_epoch = []
#        for epoch in range(num_epochs):
#            curr_loss = 0.0
#            for batch in range(n_batches):
#                x_i, y_i = x[batch*batch_size:(batch+1)*batch_size], y[batch*batch_size:(batch+1)*batch_size]

#                #Forward pass
#                pred_i = self.forward(x_i)

#                #Calculate the loss
#                loss = criterion(pred_i.squeeze(1), y_i.float())
#                #Zero gradients
#                optimizer.zero_grad()

#                #Compute gradients
#                loss.backward()
#                optimizer.step()

#                if curr_loss <= loss:
#                    curr_loss = loss
#            #Log pass
#            print('Epoch: %03d' % (epoch + 1), end="")
#            print(' | Train ACC: %.2f' % self.evaluate(x_i, y_i), end="%")
#            print(' | Cost: %.3f' % curr_loss)
#            cost_per_epoch.append(curr_loss.cpu().detach().numpy())


#        return cost_per_epoch


#net2 = Logistic_Regression2(x_data_normalized.size(1))
#loss2 = net2.train_pass(x_data_normalized, y_data_normalized, 300, 0.01)
##loss2 = net2.train_pass_sgd(x_data_normalized, y_data_normalized, 300, 0.05)
##loss2 = net2.train_pass_mb(x_data_normalized, y_data_normalized, 3000, 0.001)

## summarize history for loss
#fig, axs = plt.subplots(figsize = (40,6))
#plt.plot(loss2)
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()






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

data_cancer = load_breast_cancer()
print(data_cancer.keys())

x = data_cancer['data']
y = data_cancer['target']
names = data_cancer['target_names']
feature_names = data_cancer['feature_names']

x_data = torch.from_numpy(x).to(dtype=torch.float32, device=device)
print("X shape: ", x_data.shape)

y_data = torch.from_numpy(y).to(dtype=torch.int32, device=device)
print("Y shape: ", y_data.shape)


x_train, y_train = x_data[:, :2].clone(), y_data.clone()

print(x_train.shape)
print(y_train.shape)


x_train_normalized = data_normalization(x_train)
print(y_train)


'''
    Manual Gradients
'''
def BCE_loss(pred, target):
    return -(torch.log(pred) * target + (1. - target) * torch.log(1. - pred)).mean()


class Logistic_Regression_Manual(nn.Module):
    def __init__(self, num_features:torch.int32=1):
        super().__init__()
        self.num_features = num_features
        self.weights = torch.rand((1, num_features), dtype=torch.float32, device=device)
        self.bias = torch.ones(1, dtype=torch.float32, device=device)

    def sigmoid(self, x):
        return 1. / (1. + torch.exp(-x))

    def forward(self, x):
        output = torch.sum(torch.mul(x, self.weights), dim=1) + self.bias
        output = self.sigmoid(output)
        return output

    def loss(self, pred, target):
        return BCE_loss(pred, target)

    def backward(self, x, pred, target):
        pred_minus_target = torch.subtract(pred, target)
        dw = torch.sum(torch.mul(pred_minus_target.unsqueeze(1), x))
        db = torch.sum(pred_minus_target)
        return dw, db

    def predict(self, x):
        pred = self.forward(x)
        labels = torch.where(pred < 0.5, 0, 1)
        return labels

    def evaluate(self, x, y):
        labels = self.predict(x)
        if y.dim() == 0:
            y = y.unsqueeze(0)
        accuracy = (torch.sum(torch.eq(labels, y)).float() / y.size(0)) * 100.
        return accuracy

    def train_pass(self, x, y, 
                   num_epochs:torch.int32=10,
                   lr:torch.float32=0.001):

        cost_per_epoch = []
        for epoch in range(num_epochs):

            #Forward pass
            pred = self.forward(x)

            #Calculate the loss
            loss = self.loss(pred, y)
            cost_per_epoch.append(loss.cpu())

            #Compute gradients
            grad_w, grad_b = self.backward(x, pred, y)

            #Update weights
            self.weights = self.weights - lr * grad_w
            self.bias = self.bias - lr * grad_b

            #Log pass
            print('Epoch: %03d' % (epoch + 1), end="")
            print(' | Train ACC: %.2f' % self.evaluate(x, y), end="%")
            print(' | Cost: %.3f' % loss)

        return cost_per_epoch

    def train_pass_sgd(self, x, y, 
                   num_epochs:torch.int32=10,
                   lr:torch.float32=0.001):

        cost_per_epoch = []
        for epoch in range(num_epochs):
            #loss = 0
            change_w = 0.0
            change_b = 0.0
            for x_i, y_i in zip(x, y):
                #Forward pass
                pred = self.forward(x_i)

                #Calculate the loss
                loss = self.loss(pred, y_i)
                cost_per_epoch.append(loss.cpu())

                #Compute gradients
                grad_w, grad_b = self.backward(x_i, pred, y_i)

                ##Update weights
                self.weights = self.weights - lr * grad_w
                self.bias = self.bias - lr * grad_b

                #Log pass
                print('Epoch: %03d' % (epoch + 1), end="")
                print(' | Train ACC: %.2f' % self.evaluate(x, y), end="%")
                print(' | Cost: %.3f' % loss)

        return cost_per_epoch


net = Logistic_Regression_Manual(x_train_normalized.size(1))
loss = net.train_pass(x_train_normalized, y_train, 300, 0.001)
#loss = net.train_pass_sgd(x_train_normalized, y_train, 300, 0.05)


# summarize history for loss
fig, axs = plt.subplots(figsize = (20,6))
plt.plot(loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


'''
    Autograd Gradients
'''

class Logistic_Regression_Auto(nn.Module):
    def __init__(self, num_features:torch.int32=1):
        super().__init__()
        self.linear = nn.Linear(num_features, 1, device=device)

    def forward(self, x):
        output = torch.sigmoid(self.linear(x))
        return output

    def predict(self, x):
        pred = self.forward(x)
        labels = torch.where(pred < 0.5, 0, 1)
        return labels

    def evaluate(self, x, y):
        labels = self.predict(x)
        if y.dim() == 0:
            y = y.unsqueeze(0)
        accuracy = torch.sum(torch.eq(labels.squeeze(1), y)).float() / y.size(0)
        return accuracy

    def train_pass(self, x, y, 
                   num_epochs:torch.int32=10,
                   lr:torch.float32=0.001):

        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        cost_per_epoch = []
        for epoch in range(num_epochs):

            #Forward pass
            pred = self.forward(x)

            #Calculate the loss
            loss = criterion(pred.squeeze(1), y.float())
            cost_per_epoch.append(loss.cpu().detach().numpy())

            #Compute gradients
            loss.backward()
            optimizer.step()

            #Zero gradients
            optimizer.zero_grad()
                
            #Log pass
            print('Epoch: %03d' % (epoch + 1), end="")
            print(' | Train ACC: %.2f' % self.evaluate(x, y), end="%")
            print(' | Cost: %.3f' % loss)

        return cost_per_epoch

    def train_pass_sgd(self, x, y, 
                num_epochs:torch.int32=10,
                lr:torch.float32=0.001):

        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.02)

        cost_per_epoch = []
        for epoch in range(num_epochs):
            for x_i, y_i in zip(x, y):
                #Forward pass
                pred_i = self.forward(x_i)

                #Calculate the loss
                loss = criterion(pred_i.squeeze(0), y_i.float())
                cost_per_epoch.append(loss.cpu().detach().numpy())

                #Zero gradients
                optimizer.zero_grad()

                #Compute gradients
                loss.backward()
                optimizer.step()

                #Log pass
                print('Epoch: %03d' % (epoch + 1), end="")
                print(' | Train ACC: %.2f' % self.evaluate(x_i, y_i), end="%")
                print(' | Cost: %.3f' % loss)

        return cost_per_epoch

    def train_pass_mb(self, x, y, 
                num_epochs:torch.int32=10,
                lr:torch.float32=0.001):

        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        batch_size = 100
        n_batches = x.size(0) // batch_size

        cost_per_epoch = []
        for epoch in range(num_epochs):
            for batch in range(n_batches):
                x_i, y_i = x[batch*batch_size:(batch+1)*batch_size], y[batch*batch_size:(batch+1)*batch_size]

                #Forward pass
                pred_i = self.forward(x_i)

                #Calculate the loss
                loss = criterion(pred_i.squeeze(1), y_i.float())
                cost_per_epoch.append(loss.cpu().detach().numpy())

                #Zero gradients
                optimizer.zero_grad()

                #Compute gradients
                loss.backward()
                optimizer.step()

                #Log pass
                print('Epoch: %03d' % (epoch + 1), end="")
                print(' | Train ACC: %.2f' % self.evaluate(x_i, y_i), end="%")
                print(' | Cost: %.3f' % loss)


        return cost_per_epoch


net2 = Logistic_Regression_Auto(x_train_normalized.size(1))
loss2 = net2.train_pass(x_train_normalized, y_train, 1500, 0.01)
#loss2 = net2.train_pass_sgd(x_train_normalized, y_train, 300, 0.05)
#loss2 = net2.train_pass_mb(x_train_normalized, y_train, 3000, 0.001)

# summarize history for loss
fig, axs = plt.subplots(figsize = (40,6))
plt.plot(loss2)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
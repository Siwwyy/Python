#import torch
#import matplotlib.pyplot as plt

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#from sklearn.datasets import load_breast_cancer

##Get dataset
#cancer = load_breast_cancer()
#x = cancer['data']
#y = cancer['target']
#names = cancer['target_names']
#feature_names = cancer['feature_names']

#x_data = torch.from_numpy(x).to(dtype=torch.float32, device=device)
#print(x_data.shape)

#y_data = torch.from_numpy(y).to(dtype=torch.int32, device=device)
#print(y_data.shape)

#y_data

## fig, ax1 = plt.subplots(1, 1, figsize=(26, 16))
## point_color = ['red', 'darkcyan', 'darkorange']
## for idx, data in enumerate(x_data.cpu()):

##     color = point_color[y_data[idx]]

##     ax1.plot(data[0], data[1],
##              linestyle='none',
##              marker='o', color=color)
## ax1.set_xlabel(feature_names[0])
## ax1.set_ylabel(feature_names[1])
## ax1.axis('equal')
## ax1.legend()



## test_data = torch.rand(40,40)
#uniform_distribution = torch.distributions.uniform.Uniform(0.0, 5.0)
#test_data = uniform_distribution.sample((40,100))
#test_data.shape

#def plot_xy(data:torch.tensor):
#  fig, ax1 = plt.subplots(1, 1, figsize=(26, 16))

#  ax1.spines['left'].set_position('center')
#  ax1.spines['bottom'].set_position('center')
#  ax1.spines['right'].set_color('none')
#  ax1.spines['top'].set_color('none')

#  ax1.set_xlim([-7., 7.])
#  ax1.set_ylim([-7., 7.])
#  ax1.plot(data[0], data[1],
#              linestyle='none',
#              marker='o', color='red')

#plot_xy(test_data)

#'''
#  Data normalization
#  https://www.baeldung.com/cs/normalizing-inputs-artificial-neural-network
#'''
#mu, sigma = test_data.mean(), test_data.std()
#print(mu, '\n' ,sigma)
#test_data_normalized = (test_data - mu) / sigma

#mu_normalized, sigma_normalized = test_data_normalized.mean(),
#test_data_normalized.std()
#assert torch.isclose(sigma_normalized,
#torch.ones(1).to(test_data_normalized)), "SD has to be equal 1.  after
#normalization!"

#mu1, sigma1 = test_data_normalized.mean(), test_data_normalized.std()
#print(mu1, sigma1)

#plot_xy(test_data_normalized)

#def data_normalization(data:torch.tensor) -> torch.tensor:
#  mu, sigma = data.mean(), data.std()
#  return (data - mu) / sigma

#class Logistic_Regression():
#  def __init__(self, num_features):
#    self.num_features = num_features
#    self.weights = torch.rand((1, num_features), dtype=torch.float32,
#    device=device)
#    self.bias = torch.ones(1, dtype=torch.float32, device=device)

#  def sigmoid(self, x):
#    return 1 / (1 + torch.exp(-x))

#  def d_sigmoid(self, network_pred):
#    return network_pred * (1.0 - network_pred)

#  def forward(self, x):
#    network_sigma = torch.add(torch.mm(self.weights, x.t()), self.bias)
#    network_pred = self.sigmoid(network_sigma)
#    return network_pred

#  def loss(self, y, network_pred):
#    loss_value = -y * torch.log(network_pred) + (1.0 - y) * torch.log(1.0 -
#    network_pred)
#    return loss_value.sum()

#  def backward(self, x, y, network_pred):
#    # grad_wrt_pred = (network_pred - y) / (network_pred - network_pred**2)
#    grad_wrt_sigma = network_pred - y
#    grad_wrt_w = torch.mm(grad_wrt_sigma, x)
#    grad_wrt_b = torch.sum(grad_wrt_sigma)
#    return grad_wrt_w, grad_wrt_b

#  def predict_labels(self, x):
#    network_pred = self.forward(x)
#    labels = torch.where(network_pred >= .5, 1., 0.) # threshold function
#    return labels

#  def evaluate(self, x, y):
#    labels = self.predict_labels(x).float()
#    accuracy = torch.sum(labels == y.float())
#    return accuracy.float().mean()

#  def train(self, x, y, num_epochs, learning_rate=0.01):
#    epoch_cost = []
#    for e in range(2):
        
#        #### Compute outputs ####
#        output = self.forward(x)

#        #### Compute gradients ####
#        grad_w, grad_b = self.backward(x, y, output)

#        #### Update weights ####
#        self.weights -= learning_rate * grad_w
#        self.bias -= learning_rate * grad_b
        
#        # #### Logging ####
#        cost = self.loss(y, output)
#        print(cost)
#        print('Epoch: %03d' % (e+1), end="")
#        print(' | Train ACC: %.3f' % self.evaluate(x, y), end="\n")
#        # print(' | Cost: %.3f' % cost.sum())
#        # epoch_cost.append(cost.sum())
#    #return epoch_cost

#X_train_tensor = data_normalization(x_data)
#y_train_tensor = y_data.clone()

#model1 = Logistic_Regression(num_features=X_train_tensor.size(1))

#epoch_cost = model1.train(X_train_tensor, y_train_tensor, num_epochs=30,
#learning_rate=0.1)





































#print('\nModel parameters:')
#print(' Weights: %s' % model1.weights)
#print(' Bias: %s' % model1.bias)





#def plot_xydwa(data:torch.tensor):
#  fig, ax1 = plt.subplots(1, 1, figsize=(26, 16))

#  ax1.spines['left'].set_position('center')
#  ax1.spines['bottom'].set_position('center')
#  ax1.spines['right'].set_color('none')
#  ax1.spines['top'].set_color('none')

#  ax1.set_xlim([-7., 7.])
#  ax1.set_ylim([-7., 7.])
#  ax1.plot(data[:, 0], data[:, 1],
#              linestyle='none',
#              marker='o', color='red')

#y_data.shape

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from sklearn.datasets import load_breast_cancer

#Get dataset
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

#print(names)
#print(feature_names)
#print(data_cancer.DESCR)

#print(y_data)
'''
  Data normalization
  https://www.baeldung.com/cs/normalizing-inputs-artificial-neural-network
'''

def data_normalization(data:torch.tensor) -> torch.tensor:
  mu, sigma = data.mean(), data.std()
  return (data - mu) / sigma



def BCE_loss(pred, target):
    return -(torch.log(pred) * target + (1. - target) * torch.log(1. - pred)).mean()


class Logistic_Regression(nn.Module):
    def __init__(self, num_features:torch.int32=1):
        super(Logistic_Regression, self).__init__()
        self.num_features = num_features
        #self.weights = torch.rand((1, num_features), dtype=torch.float32, device=device, requires_grad=True)
        #self.bias = torch.ones(1, dtype=torch.float32, device=device, requires_grad=True)

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
        labels = torch.where(pred >= 0.5, 1, 0)
        return labels

    def evaluate(self, x, y):
        labels = self.predict(x)
        accuracy = torch.sum(torch.eq(labels, y)).float().mean()
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

            #Compute gradients
            grad_w, grad_b = self.backward(x, pred, y)

            #Update weights
            self.weights = self.weights - lr * grad_w
            self.bias = self.bias - lr * grad_b

            ## calculate gradients = backward pass
            #loss.backward()

            ## update weights
            ##w.data = w.data - learning_rate * w.grad
            #with torch.no_grad():
            #    self.weights -= lr * self.weights.grad
    
            ## zero the gradients after updating
            #self.weights.grad.zero_()

            #if epoch % 200:
            #    lr = lr + 0.001
            #Log pass
            print('Epoch: %03d' % (epoch + 1), end="")
            print(' | Train ACC: %.3f' % self.evaluate(x, y), end="")
            print(' | Cost: %.3f' % loss)
            cost_per_epoch.append(loss.cpu())

        return cost_per_epoch



amount_of_samples = 100
x_data_normalized = data_normalization(x_data)[:amount_of_samples, :2]

net = Logistic_Regression(x_data_normalized.size(1))
loss = net.train_pass(x_data_normalized, y_data[:amount_of_samples], 100, 0.08)


# summarize history for loss
plt.plot(loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
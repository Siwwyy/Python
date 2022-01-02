
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from functools import partial

from sklearn.datasets import load_digits

#Get dataset
n_class = 8
data_mnist = load_digits(n_class=n_class)
print(data_mnist.keys())

x = data_mnist['data']
y = data_mnist['target']
names = data_mnist['target_names']
feature_names = data_mnist['feature_names']

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

def data_normalization(data:torch.tensor, mean_zero:bool=False) -> torch.tensor:
    mu, sigma = data.mean(), data.std()
    return (data - 0.) / sigma if mean_zero else (data - mu) / sigma

# pick a sample to plot
#img = x_data[0].view(8,8).cpu()
#fig = plt.figure
#plt.matshow(img)
#plt.show()


#num_row = 2
#num_col = 5# plot images
#fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
#for i in range(10):
#    ax = axes[i//num_col, i%num_col]
#    img = x_data[i].view(8,8).cpu()
#    ax.imshow(img, cmap='gray')
#    ax.set_title('Label: {0}'.format(y_data[i]))
#plt.tight_layout()
#plt.show()



#x_train_normalized = data_normalization(x_data)
x_train_normalized = data_normalization(x_data, True)
#x_train_normalized = x_data.clone()
print(x_train_normalized.std())
print(x_train_normalized.mean())
print(x_train_normalized.shape)

#assert torch.isclose(x_train_normalized.std(), torch.ones(1).to(x_train_normalized)), "Standard deviation has to be ~1. after normalization"
#assert torch.isclose(x_train_normalized.mean(), torch.zeros(1).to(x_train_normalized), atol=0.1), "Mean has to be ~0. after normalization"

def plot_xy(data:torch.tensor):
    fig, ax1 = plt.subplots(1, 1, figsize=(26, 16))

    ax1.spines['left'].set_position('center')
    ax1.spines['bottom'].set_position('center')
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')

    ax1.set_xlim([-7., 7.])
    ax1.set_ylim([-7., 7.])
    ax1.plot(data[0], data[1], 
                linestyle='none', 
                marker='o', color='red')
    plt.show()


#plot_xy(x_data[:, 0:2].cpu())
#plot_xy(x_train_normalized[:, 0:2].cpu())

def to_onehot(tens:torch.tensor, num_classes:torch.int32) -> torch.tensor:
    tens_onehot = torch.zeros((tens.shape[-1], num_classes)).to(tens)
    indices = tens.long().unsqueeze(-1)
    src = 1
    dim = 1
    return tens_onehot.scatter_(dim, indices, src)

def softmax(tens:torch.tensor, dim=0) -> torch.tensor:
    shift = tens - torch.max(tens)
    return shift.exp() / torch.sum(shift.exp(), dim=dim, keepdim=True)

def cross_entropy(y_hat:torch.tensor, y:torch.tensor):
    return -(torch.sum(y * torch.log(y_hat), dim=1))

#class Softmax_Regression(nn.Module):
#    def __init__(self, input_features:int=2, num_classes:int=3):
#        super().__init__()
#        self.input_features = input_features
#        '''
#            Input -> Hidden

#            Input: 8 neurons
#            Hidden: 8x8 neurons
#            8 input neurons -> 8x8 hidden neurons
#        '''
#        self.weights_ih = torch.rand((input_features, input_features), device=device)    
#        self.bias_ih = torch.rand((input_features), device=device)
#        # Define sigmoid activation
#        self.sigmoid = nn.Sigmoid()
#        '''
#            Hidden -> Output

#            Hidden: 8x8 neurons
#            Output: 4 neurons (== num_classes)
#            8x8 hidden neurons -> 4 output neurons
#        '''
#        self.weights_ho = torch.rand((input_features, num_classes), device=device)
#        self.bias_ho = torch.rand((num_classes), device=device)
#        self.softmax = partial(softmax)

#    def forward(self, x) -> torch.tensor:
#        #input -> hidden
#        output = torch.matmul(x, self.weights_ih) + self.bias_ih
#        output = self.sigmoid(output)
#        #hidden -> output
#        output = torch.matmul(output, self.weights_ho) + self.bias_ho
#        return self.softmax(output, dim=1), output

#    def backward(self, x, y_hat, target, before_softmax) -> torch.tensor:
#        #loss -> softmax
#        d_L_yhat = y_hat - target #where: y_hat is softmax output and target is onehot GT

#        #Maybe it can be done better, but currently lets leave it in iterative way
#        jacobian_softmax = torch.zeros((y_hat.size(0), y_hat.size(1), y_hat.size(1)))
#        for i in range(y_hat.size(1)):
#            for j in range(y_hat.size(1)):
#                y_hat_i = y_hat[:, i]
#                y_hat_j = y_hat[:, j]
#                if i == j:
#                    jacobian_softmax[:, i, j] = y_hat_i * (1. - y_hat_j)
#                else:
#                    jacobian_softmax[:, i, j] = y_hat_i * (1. - y_hat_j)

#        #weights hidden -> output
#        grad_weights_ho = d_L_yhat * jacobian_softmax * before_softmax
#        #weights input -> hidden

#    def loss(self, y_hat, target) -> torch.tensor:
#        return cross_entropy(y_hat, target)

#    def predict(self, x):
#        logits, probas = self.forward(x)
#        labels = torch.argmax(probas, dim=1)
#        return labels

#    def evaluate(self, x, y):
#        labels = self.predict(x)
#        if y.dim() == 0:
#            y = y.unsqueeze(0)
#        accuracy = (torch.sum(torch.eq(labels, y)).float() / y.size(0)) * 100.
#        return accuracy

#    def train_pass(self, x, y, num_epochs:int=1, lr:float=0.001):
        
#        cost_per_epoch = []
#        for epoch in range(num_epochs):

#            #forward pass
#            y_hat, before_softmax = self.forward(x)

#            #Calculate the loss
#            y_data_onehot = to_onehot(y_data, num_classes=y_hat.size(1))
#            loss = self.loss(y_hat, y_data_onehot)
#            cost_per_epoch.append(loss.cpu())

#            #Compute gradients
#            grad_w, grad_b = self.backward(x, y_hat, y_data_onehot, before_softmax)

#            #Update weights
#            self.weights = self.weights - lr * grad_w
#            self.bias = self.bias - lr * grad_b

#            #Log pass
#            print('Epoch: %03d' % (epoch + 1), end="")
#            print(' | Train ACC: %.2f' % self.evaluate(x, y), end="%")
#            print(' | Cost: %.3f' % loss)

#        return cost_per_epoch



class Softmax_Regression(nn.Module):
    def __init__(self, input_features:int=2, num_classes:int=3):
        super().__init__()
        self.input_features = input_features
        '''
            Input -> Hidden

            Input: 8 neurons
            Hidden: 8x8 neurons
            8 input neurons -> 8x8 hidden neurons
        '''
        self.layer_input_hidden = torch.nn.Linear(in_features=input_features,
                                              out_features=input_features*input_features, 
                                              bias=True, 
                                              device=device)
        # Define sigmoid activation
        self.sigmoid = nn.Sigmoid()
        '''
            Hidden -> Output

            Hidden: 8x8 neurons
            Output: 4 neurons (== num_classes)
            8x8 hidden neurons -> 4 output neurons
        '''
        self.layer_hidden_output = torch.nn.Linear(in_features=input_features*input_features, 
                                              out_features=num_classes, 
                                              bias=True, 
                                              device=device)

    def forward(self, x) -> torch.tensor:
        #input -> hidden
        output = self.layer_input_hidden(x)
        #hidden neurons activation
        output = self.sigmoid(output)
        #hidden -> output
        output = self.layer_hidden_output(output)
        #output == 4 classes, apply softmax
        return output

    def predict(self, x):
        probas = self.forward(x)
        labels = torch.argmax(probas, dim=1)
        return labels

    def evaluate(self, x, y):
        labels = self.predict(x).float()
        if y.dim() == 0:
            y = y.unsqueeze(0)
        accuracy = (torch.sum(torch.eq(labels, y)) / y.size(0)) * 100.
        return accuracy

    def train_pass(self, x, y, num_epochs:int=1, lr:float=0.001):
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        cost_per_epoch = []
        for epoch in range(num_epochs):

            #Zero gradients
            optimizer.zero_grad()

            #forward pass
            y_hat = self.forward(x)

            #Calculate the loss
            loss = criterion(y_hat, y.to(torch.int64))
            cost_per_epoch.append(loss.cpu().detach().numpy())

            #Compute gradients
            loss.backward()
            optimizer.step()

            #Log pass
            print('Epoch: %03d' % (epoch + 1), end="")
            print(' | Train ACC: %.2f' % self.evaluate(x, y), end="%")
            print(' | Cost: %.3f' % loss)

        return cost_per_epoch


net = Softmax_Regression(x_train_normalized.size(1), n_class)
loss = net.train_pass(x_train_normalized[:700,...], y_data[:700], num_epochs=500, lr=0.01)

# summarize history for loss
fig, axs = plt.subplots(figsize = (20,6))
plt.plot(loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



# Prediction
test_data = x_train_normalized[700:,...]
test_label = y_data[700:]

print(test_label[0])

num_row = 2
num_col = 5# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(10):
    ax = axes[i//num_col, i%num_col]
    img = test_data[i].view(8,8).cpu()
    ax.imshow(img, cmap='gray')
    ax.set_title('Label: {0}'.format(net.predict(test_data[i].unsqueeze(0)).item()))
plt.tight_layout()
plt.show()


#my_data = torch.tensor([1,2,3], dtype=torch.float32).repeat(3, 1)
#weights = torch.tensor([
#    [1, 1],
#    [2, 2],
#    [3, 3]
#    ], dtype=torch.float32)
#print(weights.shape)
#linear = nn.Linear(3, 2, bias=False)
#out = torch.matmul(my_data, weights)
#print(out)

#output_linear = linear(my_data)
#print(output_linear)
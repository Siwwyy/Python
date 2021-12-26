
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from sklearn.datasets import load_digits

#Get dataset
n_class = 4
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

def softmax(tens:torch.tensor) -> torch.tensor:
    return tens.exp() / torch.sum(tens.exp())

def cross_entropy(y_hat:torch.tensor, y:torch.tensor):
    return -(torch.sum(y * torch.log(y_hat)))

class Softmax_Regression(nn.Module):
    def __init__(self, input_features:int=2, num_classes:int=3):
        self.input_features = input_features

        '''
            Input -> Hidden

            Input: 8 neurons
            Hidden: 8x8 neurons
            8 input neurons -> 8x8 hidden neurons
        '''
        self.weights_ih = torch.rand((input_features, input_features), device=device)    
        self.bias_ih = torch.rand((input_features), device=device)
        '''
            Hidden -> Output

            Hidden: 8x8 neurons
            Output: 4 neurons (== num_classes)
            8x8 hidden neurons -> 4 output neurons
        '''
        self.weights_ho = torch.rand((input_features, num_classes), device=device)    
        self.bias_ho = torch.rand((num_classes), device=device)

    def forward(self, x) -> torch.tensor:
        #input -> hidden
        output = torch.matmul(x[..., :30], self.weights_ih[:30, :2])# + self.bias_ih
        print(output.shape)

        linear = nn.Linear(64, 2, device=device)
        output_linear = linear(x)
        print(output_linear)

        #hidden -> output
        output = torch.matmul(output, self.weights_ho) + self.bias_ho
        print(output)
        return softmax(output)

    def backward(self, x, y_hat, target) -> torch.tensor:
        pass

    def train_pass(self, x, y, num_epochs:int=1, lr:float=0.001):
        
        cost_per_epoch = []
        for epoch in range(num_epochs):

            #forward pass
            y_hat = self.forward(x)

            print(y_hat)
            break

            #Log pass
            print('Epoch: %03d' % (epoch + 1), end="")
            #print(' | Train ACC: %.2f' % self.evaluate(x, y), end="%")
            #print(' | Cost: %.3f' % loss)

        return cost_per_epoch


net = Softmax_Regression(x_train_normalized.size(1), n_class)
loss = net.train_pass(x_train_normalized, y_data, num_epochs=200, lr=0.001)

## summarize history for loss
#fig, axs = plt.subplots(figsize = (20,6))
#plt.plot(loss)
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from sklearn.datasets import load_digits

#Get dataset
data_mnist = load_digits(n_class=4)
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

def data_normalization(data:torch.tensor) -> torch.tensor:
  mu, sigma = data.mean(), data.std()
  return (data - mu) / sigma

# pick a sample to plot
#img = x_data[0].view(8,8).cpu()
#fig = plt.figure
#plt.matshow(img)
#plt.show()


num_row = 2
num_col = 5# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(10):
    ax = axes[i//num_col, i%num_col]
    img = x_data[i].view(8,8).cpu()
    ax.imshow(img, cmap='gray')
    ax.set_title('Label: {0}'.format(y_data[i]))
plt.tight_layout()
plt.show()





class Softmax_Regression(nn.Module):
    def __init__(self, num_classes:int=2):
        pass





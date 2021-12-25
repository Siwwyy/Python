
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

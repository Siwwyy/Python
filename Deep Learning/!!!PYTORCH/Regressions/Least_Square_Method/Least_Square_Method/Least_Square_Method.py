import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#Least Square method
# https://web.williams.edu/Mathematics/sjmiller/public_html/BrownClasses/54/handouts/MethodLeastSquares.pdf
def lsm(x_points, y_points):
  curr_dtype = torch.float32
  #inverse matrix
  x_matrix = torch.tensor([
      [  torch.sum(torch.pow(x_points, 2)), torch.sum(x_points) ], # 0,0 | 0,1
      [  torch.sum(x_points), torch.sum(torch.ones(len(x_points)))  ]                     # 1,0 | 1,1
    ], dtype=curr_dtype)  

  inv_matrix = torch.inverse(x_matrix)

  det_x_matrix = torch.det(inv_matrix)
  assert det_x_matrix != torch.zeros(1), "Determinant has to not be equal to 1"

  #y matrix
  y_matrix = torch.tensor([
      [  torch.sum(torch.mul(x_points, y_points)) ], # 0,0
      [  torch.sum(y_points) ]                       # 1,0
      ], dtype=curr_dtype)

  #find a and b coefficients y = ax + b
  return torch.mm(inv_matrix, y_matrix)  


import numpy as np
#EXERCISE
number_of_points = 1000
x_point = []
y_point = []

a = 0.22
b = 0.78


for i in range(number_of_points):
    x = np.random.normal(0.0,0.5)
    y = (a*x+b)+np.random.normal(0.0,0.1)
    x_point.append(x)
    y_point.append(y)


plt.scatter(x_point,y_point,c='b')
plt.show()


x_p = torch.tensor(x_point)
y_p = torch.tensor(y_point)


# x_p = torch.tensor([1, 2, 3, 4, 5, 6, 7])
# y_p = torch.tensor([1.5, 3.8, 6.7, 9.0, 11.2, 13.6, 16])

a_b = lsm(x_p, y_p)


foo1 = a_b[0] * x_p + a_b[1]


# Assign variables to the y axis part of the curve

plt.figure(figsize=(15,15))
# Plotting both the curves simultaneously
plt.scatter(x_p, y_p, color='r')
plt.plot(x_p, foo1, color='g')
  
# To load the display window
plt.show()
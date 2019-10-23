# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 18:21:26 2019

@author: nrgurram
"""

import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F

from itertools import count

data = pd.read_csv("./qn2_data.csv", header=None)
batch_size = 5
  
'''
 Create a linear model with X*W + b with two input features and one output
'''
model = torch.nn.Linear(in_features=2, out_features=1, bias=True)

# Update the parameters with SGD with learning rate as 0.1
sgd = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in count(1):
  
  model.zero_grad()
  
  rand_indices = np.random.randint(0, data.shape[0], size=batch_size)
  batch_data = data.iloc[rand_indices, :]
  X = torch.tensor(batch_data.iloc[:, [0,1]].values, dtype=torch.float)
  y = torch.tensor(batch_data.iloc[:, -1].values, dtype=torch.float)
  y = y.unsqueeze(1)
  
  # Predict MSE loss
  loss = F.smooth_l1_loss(model(X), y)
  
  # Propagate gradient
  loss.backward()
  
  sgd.step()
  
  if loss.item() < 0.05:
    break

print('Loss: {:.6f} after {} batches'.format(loss.item(), epoch))
print(" Learned  parameters : w1 {:+.2f}, w2 {:+.2f} bias {:+.2f} " .format(model.weight.view(-1)[0], model.weight.view(-1)[1], model.bias[0]))

# Predict fir the given test data
test_set = torch.tensor([[6,4],[10,5],[14,8]] , dtype=torch.float)
predictions = model(test_set)
print("Test predictions \n")
print(predictions)
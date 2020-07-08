import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from torch.autograd import Variable

from utils import train, test, load_data
from model import RNNModel

import warnings
warnings.filterwarnings("ignore")

seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

print("Reading CSV file")
df = pd.read_csv("data/prices-split-adjusted.csv", index_col = 0)
df_stock = df[df.symbol == 'EQIX'].copy()
df_stock.drop(['symbol'],1,inplace=True)
df_stock.drop(['volume'],1,inplace=True)
df_stock = df_stock.copy()

seq_len = 50 
valid_set_size_percentage = 10 
test_set_size_percentage = 10 
print("Making train-validation-test data")
x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(df_stock, seq_len, valid_set_size_percentage, test_set_size_percentage)

batch_size = 50
n_epochs = 50 

tensor_x = torch.Tensor(x_train) 
tensor_y = torch.Tensor(y_train) 
my_dataset = TensorDataset(tensor_x,tensor_y) 
dataloader_train = DataLoader(my_dataset, batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=False) 

tensor_x = torch.Tensor(x_valid) 
tensor_y = torch.Tensor(y_valid) 
my_dataset = TensorDataset(tensor_x,tensor_y) 
dataloader_val = DataLoader(my_dataset, batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=False) 

tensor_x = torch.Tensor(x_test) 
tensor_y = torch.Tensor(y_test) 
my_dataset = TensorDataset(tensor_x,tensor_y) 
dataloader_test = DataLoader(my_dataset, batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=False) 

hidden_dim = 200
num_layers = 4
lr = 1e-3
log_dir = './ckpt'
model_name = 'model.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Found {device} ...")
print("Instantiating RNN Model")

if not os.path.exists(log_dir):
  os.mkdir(log_dir)
model_save_path = os.path.join(log_dir,model_name)
model = RNNModel(x_train.shape[-1],hidden_dim,num_layers,y_train.shape[-1]).to(device)
optimizer = optim.Adam(model.parameters(),lr=lr)
criterion = nn.MSELoss()

print("< Training starts >")
model = train(model,dataloader_train,dataloader_val,device,criterion,optimizer,n_epochs,model_save_path)


print("Testing on test data-set ")
log_dir = './ckpt'
model_name = 'model.pth'
model_save_path = os.path.join(log_dir,model_name)
output_dim = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = RNNModel(x_test.shape[-1],hidden_dim,num_layers,output_dim).to(device)
y_test_pred = test(x_test,model,model_save_path,device)
ft_dict = {
          0: 'open',
          1: 'high',
          2: 'low',
          3: 'close', 
}
fig = plt.figure(figsize=(15,10))
for i in range(4):
  n = len(y_test_pred)
  ax = fig.add_subplot(2,2,i+1)
  ax.plot(range(n),y_test[:,i],range(n),y_test_pred[:,i])
  ax.legend(['test','test_predicted'])
  ax.set_ylabel('Price')
  ax.set_xlabel('Time (days)')
  ax.set_title(f'Prediction of future stock prices - {ft_dict[i]} category - (on test-set)')
plt.show()



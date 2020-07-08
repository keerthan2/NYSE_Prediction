import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from model import RNNModel

import warnings
warnings.filterwarnings("ignore")

seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def train(model,dataloader_train,dataloader_val,device,criterion,optimizer,n_epochs,model_save_path):
  hist_loss = np.zeros(n_epochs)
  hist_loss_val = np.zeros(n_epochs)
  val_loss_best = np.inf
  for idx_epoch in range(n_epochs):
      running_loss = 0
      with tqdm(total=len(dataloader_train.dataset), desc=f"[Epoch {idx_epoch+1:3d}/{n_epochs}]") as pbar:
          for idx_batch, (x, y) in enumerate(dataloader_train):
              optimizer.zero_grad()
              netout = model(x.to(device))
              
              loss = criterion(netout,y.to(device))
              loss.backward()
              optimizer.step()

              running_loss += loss.item()
              pbar.set_postfix({'loss': running_loss/(idx_batch+1)})
              pbar.update(x.shape[0])
          
          train_loss = running_loss/len(dataloader_train)
          val_running_loss = 0
          with torch.no_grad():
            for x, y in dataloader_val:
                netout = model(x.to(device))
                val_running_loss += criterion(y, netout).item()
          val_loss = val_running_loss/len(dataloader_val)
          pbar.set_postfix({'loss': train_loss, 'val_loss': val_loss})
          
          hist_loss[idx_epoch] = train_loss
          hist_loss_val[idx_epoch] = val_loss
          
          if val_loss < val_loss_best:
              val_loss_best = val_loss
              best_model = model
              torch.save(model.state_dict(), model_save_path)
   
  plt.plot(hist_loss[1:], 'o-', label='train')
  plt.plot(hist_loss_val[1:], 'o-', label='val')
  plt.legend()
  plt.title('Train Loss and Validation Loss ')
  plt.ylabel('Loss')
  plt.xlabel('Epochs')
  print(f"model exported to {model_save_path} with loss {val_loss_best:5f}")

  return best_model

def test(x_test,model,model_save_path,device):
  model.load_state_dict(torch.load(model_save_path,map_location=device))
  x = torch.from_numpy(x_test).to(device).float()
  y_test_pred = model(x).cpu().detach().numpy()
  return y_test_pred

def load_data(stock, seq_len, valid_set_size_percentage, test_set_size_percentage):
    data_raw = stock.to_numpy() 
    data = []
    
    for index in range(len(data_raw) - seq_len): 
        data.append(data_raw[index: index + seq_len])
    
    data = np.array(data)
    valid_set_size = int(np.round(valid_set_size_percentage/100*data.shape[0]))  
    test_set_size = int(np.round(test_set_size_percentage/100*data.shape[0]))
    train_set_size = data.shape[0] - (valid_set_size + test_set_size)
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_valid = data[train_set_size:train_set_size+valid_set_size,:-1,:]
    y_valid = data[train_set_size:train_set_size+valid_set_size,-1,:]
    
    x_test = data[train_set_size+valid_set_size:,:-1,:]
    y_test = data[train_set_size+valid_set_size:,-1,:]
    
    return [x_train, y_train, x_valid, y_valid, x_test, y_test]

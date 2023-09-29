# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from WarpST import WarpST
from ops import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
import os
import shutil
from config import get_config

def plot(im):
  im = np.array(im.tolist())
  plt.imshow(im, cmap='gray', vmin=0, vmax=1)
  plt.show()
  return None

class CNN(nn.Module):

  def __init__(self):
    super().__init__()

    self.enc_x = nn.Sequential(
      conv2d(2, 64, 3, 1, 1, bias=False), # 64 x 28 x 28   
	  
      nn.BatchNorm2d(64, momentum=0.9, eps=1e-5),
      nn.ELU(),   
      nn.AvgPool2d(2, 2, 0), # 64 x 14 x 14
		# Exponential Linear Unit (ELU) function
		
      conv2d(64, 128, 3, 1, 1, bias=False),
      nn.BatchNorm2d(128, momentum=0.9, eps=1e-5),
      nn.ELU(),
      conv2d(128, 128, 3, 1, 1, bias=False),
      nn.BatchNorm2d(128, momentum=0.9, eps=1e-5),
      nn.ELU(),
      nn.AvgPool2d(2, 2, 0),  # 64 x 7 x 7

      conv2d(128, 2, 3, 1, 1), # 2 x 7 x 7     
      nn.Tanh()
    )

  def forward(self, x):
    x = self.enc_x(x)
    return x


class DIRNet(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.vCNN = CNN()
    self.config = config

  def forward(self, x, y, train=True):
    config = get_config(is_train=True)
    xy = torch.cat((x, y), dim = 1)  
    v = self.vCNN(xy)    
    
    z,V_size = WarpST(x, v, self.config.im_size)  
    
    z = z.permute(0, 3, 1, 2)
    metric_loss = mse(y, z)
    DVF_loss = torch.norm(v[0,:],2) + torch.norm(v[1,:],2)
    loss = config.l1_alpha*metric_loss + 100000*DVF_loss
    print("metric_loss:{},DVF_loss:{}".format(config.l1_alpha*metric_loss,50000*DVF_loss))
    if train:
      return z, loss
    else:
      return z, loss, v.detach(), V_size.detach(), x, y
	  
	  

  def deploy(self, dir_path, x, y, names=None):
    config = get_config(is_train=True) 
    with torch.no_grad():
      z, _, v , V_size, x, y = self.forward(x, y, train=False)
      for i, name in enumerate(names):  
        
        if os.path.exists(("{}\\"+config.CRCS_foldername).format(name)) == True:
          shutil.rmtree(("{}\\"+config.CRCS_foldername).format(name))
        os.mkdir(("{}\\"+config.CRCS_foldername).format(name))
        save_image_with_scale(("{}\\"+config.CRCS_foldername+"\\z.tif").format(name), z.permute(0, 2, 3, 1)[i,:,:,0].numpy())  # z的torch.Size([2, 1, 180, 256])  
        save_xy_image_with_scale(("{}\\"+config.CRCS_foldername+"\\x.tif").format(name), x.permute(0, 2, 3, 1)[i,:,:,0].numpy())
        save_xy_image_with_scale(("{}\\"+config.CRCS_foldername+"\\y.tif").format(name), y.permute(0, 2, 3, 1)[i,:,:,0].numpy())
        
        savemat(("{}\\"+config.CRCS_foldername+"\\V.mat").format(name),{"V":V_size[i,:,:,:].numpy()})

		
		
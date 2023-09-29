import torch
import random
from torch import nn
from models import DIRNet
from config import get_config
from data import SlideDataHandler
from ops import mkdir
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as tf
from torch.optim.lr_scheduler import StepLR
import numpy as np

import glob


manualSeed = random.randint(1, 10000) 
manualSeed = 2790
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# train_batch = 60000
# test_batch = 10000

def l1_regularization(model, l1_alpha):
    l1_loss = []
    for module in model.modules():
        if type(module) is nn.BatchNorm2d:
            l1_loss.append(torch.abs(module.weight).sum())
    return l1_alpha * sum(l1_loss)

def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)


			
class ParallelModels(nn.Module):

  def __init__(self, model1, model2):
    super().__init__()
    self.model1 = model1
    self.model2 = model2
  def forward(self, x):  
    out1 = self.model1(x[:,:1,:,:])    
    out2 = self.model2(x[:,1:,:,:])    
    return torch.cat([out1, out2], dim = 1)

feature_extractor_length = 3

def main():
  config = get_config(is_train=True) 
  
  model = DIRNet(config)
  if config.pretrained_layers_folder !="":
    pretrained_layers_files = glob.glob(config.pretrained_layers_folder+ r"\*netD*.model")
    assert len(pretrained_layers_files) == 2
    pretrained_models = [torch.load(pretrained_layers_files[0]),torch.load(pretrained_layers_files[1])]
    feature_extractor = ParallelModels(pretrained_models[0][:feature_extractor_length], pretrained_models[1][:feature_extractor_length])
    
    model.vCNN.enc_x[0] = nn.Sequential(feature_extractor, nn.Conv2d(64*2*2,64,1)) 
  optim = torch.optim.Adam(model.parameters(), lr = config.lr, weight_decay=0)
  scheduler = StepLR(optim, step_size=200, gamma=0.5)

  train_pr = SlideDataHandler(config.train_folder)
  test_pr = SlideDataHandler(config.test_folder)

  total_loss = 0
  for i in range(config.iteration):

    batch_x, batch_y = train_pr.sample_pair(config.batch_size)
    optim.zero_grad()
    _, loss = model(batch_x, batch_y)
    
    
    
    loss.backward()
    optim.step()
    scheduler.step()
    total_loss += loss

    if (i+1) % 100 == 0:
      print("iter {:>6d} : {}".format(i + 1, total_loss))
      total_loss = 0
      batch_x, batch_y, names = test_pr.sample_ergodic(return_name=True)
      model.deploy(config.test_folder, batch_x, batch_y, names)
      
  torch.save(model.vCNN, config.subdir+r"\\kfb_binary_initial_GAN_layer_L1_DIRNet"+ str(manualSeed) +"_size64_kernelsize3_"+str(config.iteration)+".model")
if __name__ == "__main__":
  main()


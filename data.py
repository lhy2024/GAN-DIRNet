import numpy as np
from glob import glob
import random
from PIL import Image
import torch
from torchvision.transforms import functional as F
from config import get_config
import scipy.io as io
import numpy as np

class MNISTDataHandler(object): 
  """
    Members :
      is_train - Options for sampling
      path - MNIST data path
      data - a list of np.array w/ shape [batch_size, 28, 28, 1]
  """
  def __init__(self, digit):
    self.data = digit

  def sample_pair(self, batch_size, label=None):
    label = np.random.randint(10) if label is None else label 
  
    images = self.data[label]
    
    choice1 = np.random.choice(images.shape[0], batch_size) 
    choice2 = np.random.choice(images.shape[0], batch_size)  
  
    x = images[choice1]
    y = images[choice2]

    return x, y
	

class SlideDataHandler(object):    
  def __init__(self, folder):
    if not folder.endswith("\\"):
      folder = folder+"\\"
    self.folder = folder
    self.subfolders_train = glob(folder+"??_*\\")
    self.subfolders_test = glob(folder+"??_*\\")
  def sample_pair(self, batch_size, return_name=False):
    config = get_config(is_train=True)
    choices = random.sample(self.subfolders_train, batch_size)
    fixed_list = []
    moving_list = []
    for subfolder in choices:
      print("subfolder", subfolder)
      fixed_name = glob(subfolder+"binary_fixed.png")
      assert len(fixed_name) == 1
      fixed_name = fixed_name[0]   
      moving_name = glob(subfolder+"binary_initial_moving.png")
      assert len(moving_name) == 1
      moving_name = moving_name[0]
      
     
      fill_fixed = np.loadtxt(subfolder+"equal_fixed.txt")
      fill_moving = np.loadtxt(subfolder+"equal_moving.txt")
     
        
      tform = io.loadmat(subfolder+"tform.mat")
     
      HEIGHT = config.im_size[0]  
      WIDTH = config.im_size[1]  
       
      f = Image.open(fixed_name)
      fixed_image = F.pil_to_tensor(f)  
      f.close()   
  
      m = Image.open(moving_name)
      moving_image = F.pil_to_tensor(m)
      m.close()   
      
                    
      max_frame = max(fixed_image.shape[1],fixed_image.shape[2],moving_image.shape[1],moving_image.shape[2])


      fixed_image = F.pad(fixed_image, padding=(0,0,max_frame-fixed_image.shape[2],max_frame-fixed_image.shape[1]), fill=0, padding_mode="constant") 
        
      fixed_image = F.resize(fixed_image, size=HEIGHT-1, max_size=HEIGHT)  
                    
      fixed_image = F.pad(fixed_image, padding=(0,0,WIDTH-fixed_image.shape[2],HEIGHT-fixed_image.shape[1]), padding_mode="edge")      
                   
      
     
      moving_image = F.pad(moving_image, padding=(0,0,max_frame-moving_image.shape[2],max_frame-moving_image.shape[1]), fill=0, padding_mode="constant") 
      
      moving_image = F.resize(moving_image, size=HEIGHT-1, max_size=HEIGHT) 
      moving_image = F.pad(moving_image, padding=(0,0,WIDTH-moving_image.shape[2],HEIGHT-moving_image.shape[1]), padding_mode="edge") 
                    
      

      fixed_list.append(fixed_image)
      moving_list.append(moving_image)
    fixed = torch.stack(fixed_list)  
    fixed = fixed.float() 
    moving = torch.stack(moving_list)
    moving = moving.float()
    if return_name:
      return moving, fixed, choices
    else:
      return moving, fixed
  
  def sample_ergodic(self, return_name=True):
    config = get_config(is_train=False)
    fixed_list = []
    moving_list = []
    for subfolder in self.subfolders_test:
      print("test subfolder", subfolder)
      fixed_name = glob(subfolder+"binary_fixed.png")
      assert len(fixed_name) == 1
      fixed_name = fixed_name[0]   
      moving_name = glob(subfolder+"binary_initial_moving.png")
      assert len(moving_name) == 1
      moving_name = moving_name[0]
      
      fill_fixed = np.loadtxt(subfolder+"equal_fixed.txt")
      fill_moving = np.loadtxt(subfolder+"equal_moving.txt")
      HEIGHT = config.im_size[0]
      WIDTH = config.im_size[1]
      
      f = Image.open(fixed_name)
      fixed_image = F.pil_to_tensor(f) 
      f.close()   
      m = Image.open(moving_name)
      moving_image = F.pil_to_tensor(m)
      m.close()  
      
        
      max_frame = max(fixed_image.shape[1],fixed_image.shape[2],moving_image.shape[1],moving_image.shape[2])

      fixed_frame = F.pad(fixed_image, padding=(0,0,max_frame-fixed_image.shape[2],max_frame-fixed_image.shape[1]), fill=0, padding_mode="constant") 
              
      fixed_image = F.resize(fixed_frame, size=HEIGHT-1, max_size=HEIGHT)
      fixed_image = F.pad(fixed_image, padding=(0,0,WIDTH-fixed_image.shape[2],HEIGHT-fixed_image.shape[1]), fill=0, padding_mode="constant")   
               
      moving_frame = F.pad(moving_image, padding=(0,0,max_frame-moving_image.shape[2],max_frame-moving_image.shape[1]), fill=0, padding_mode="constant") 
      moving_image = F.resize(moving_frame, size=HEIGHT-1, max_size=HEIGHT) 
      moving_image = F.pad(moving_image, padding=(0,0,WIDTH-moving_image.shape[2],HEIGHT-moving_image.shape[1]), fill=0, padding_mode="constant")

      fixed_list.append(fixed_image)
      moving_list.append(moving_image)
    fixed = torch.stack(fixed_list)  
    fixed = fixed.float() 
    moving = torch.stack(moving_list)
    moving = moving.float()
    if return_name:
      return moving, fixed, self.subfolders_test
    else:
      return moving, fixed 


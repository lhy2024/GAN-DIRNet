class Config(object):
  pass

def get_config(is_train):
  config = Config()
  config.train_folder = r"E:\2_task\dataset_CRCS"
  
  config.test_folder = r"E:\2_task\dataset_CRCS"
  
  config.foldername = "kfb_binary_initial_GAN_layer_L1_DIRNet"
  config.CRCS_foldername = "binary_initial_GAN_layer_L2_DIRNet"
  
  config.tmp_dir = "E:\\2_task\\DIRNet-PyTorch-master\\" + config.foldername + "\\tmp"
  config.ckpt_dir = "E:\\2_task\\DIRNet-PyTorch-master\\" + config.foldername + "\\ckpt"
  
  config.pretrained_layers_folder = r"E:\\2_task\\DIRNet-PyTorch-master\\" + config.foldername
  config.subdir = "E:\\2_task\\DIRNet-PyTorch-master\\" + config.foldername
  
  if is_train:
    config.batch_size = 2
    config.im_size = [180, 180]   
    config.lr = 1e-4
    config.iteration = 300
    
    config.l1_alpha = 0.00001  
    config.l2_alpha = 0.00001
    # config.tmp_dir = "tmp"
    # config.ckpt_dir = "ckpt"
  else:
    config.batch_size = 2
    config.im_size = [180, 180]

    # config.result_dir = "result"
    # config.ckpt_dir = "ckpt"
  return config

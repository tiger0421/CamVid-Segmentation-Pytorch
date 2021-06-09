import torch
import os
from torch import cuda
from src.utils import Color_map
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class config(object):

  model_path = "./U-Net/model/"
  path = "./U-Net/Dataset/"
  load_model = "./U-Net/model/state_dict.pt"
  batch = 4
  lr = 0.0001
  epochs = 40
  input_size = (128,128)
  input_size = (512,512)
  if cuda.is_available(): device = torch.device("cuda")
  else: device = torch.device('cpu')
  code2id, id2code, name2id, id2name = Color_map(path+'class_dict.csv')

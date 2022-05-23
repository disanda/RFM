import os
import numpy as np
import warnings
import torch
import pickle
import gzip
import math
import torchvision
import matplotlib.pylab as plt
import sys
sys.path.append('../')
from model.stylegan1.net import Generator #StyleGANv1

name = '001'
img_size = 1024
use_gpu = False
G_path = './checkpoint/stylegan1/ffhq1024/'
wy_path = './checkpoint/stylegan1/face_wy/01_norm.py'
wd_path = './checkpoint/stylegan1/direction_wd/id3_dict1_l1_8k_acc0.88708_saga_iter2000_0.5_1092.npy'


## Loading Pre-trained Model, Directions
Gs = Generator(startf=16, maxf=512, layer_count=int(math.log(img_size,2)-1), latent_size=512, channels=3)
Gs.load_state_dict(torch.load(G_path+'Gs_dict.pth', map_location=device))

## w_y from Real image inversion
device = torch.device("cuda" if use_gpu else "cpu")
w_y = torch.load(w_path, map_location=device).clone().squeeze(0) # face

## w_d face latent dirrections from semantic attribute: smile, eyeglasses, pose, age, gender, etc.
w_d = torch.tensor(np.load(wd_path))
#direction = direction.reshape(1,1,512).repeat(1,18,1) # z -> w , w_d = w_d / w_d.norm()

## trim w_d
layers = 0
layere = 18
w_d = w_d.view(layere-layers,512)
clip1 = 0.0 # 0.01-0.03
w_d[torch.abs(w_d)<=clip1] = 0.0

## RFM: x_d = G(x_y + a * x_d)
a= -100 #bonus   (-10) <- (-5) <- 0 ->5 ->10
start_layer= 0  #default 0, if not 0, will be bed performance
end_layer=   3  # default 3 or 4. if 3, it will keep face features (glasses). if 4, it will keep dirrection features (Smile).
w[start_layer:end_layer] = (w_y+a*w_d)[start_layer:end_layer] #w = w + bonus*direction all_w

w = w.reshape(1,18,512)
with torch.no_grad():
  x   = Gs.forward(w_y,8) # 8->1024
  x_d = Gs.forward(w,8) # 8->1024

torchvision.utils.save_image(x*0.5+0.5, './x_name'%(name))
torchvision.utils.save_image(x_d*0.5+0.5, './x_d_name'%(name))







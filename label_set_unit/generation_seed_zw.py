# random_seed_id, 生成潜空间变量(固定随机种子), 对应python 1.5.1的随机种子
import os
import math
import pickle
import numpy as np
import torch
import torchvision
from training_utils import *
from model.stylegan1.net import Generator, Mapping
import model.pggan.pggan_generator as model_pggan #PGGAN
from training_utils import *

id_s = 0
id_e = 30000

z_all = torch.zeros(id_e-id_s,512)
w_all = torch.zeros(id_e-id_s,18,512)

for i in range(id_e-id_s):
    set_seed(i+id_s)
    z = torch.randn(1,512)
    z_all[i] = z[0]
    print(i+id_s)

torch.save(z_all.clone(),'./z_%d_%d.pt'%(id_s,id_e))

##------------- synthesized seed faces -------------------
# model_type = 'StyleGANv1' # StyleGANv1, StyleGANv2, PGGAN, BigGAN
# use_gpu = False #loading stylegan model in pytorch
# img_size = 1024
# device = torch.device("cuda" if use_gpu else "cpu")

# if model_type == 'StyleGANv1':
#     model_path = '../checkpoint/stylegan1/ffhq1024/'#'../MTV-TSA/checkpoint/stylegan_v1/ffhq1024/'
#     Gs = Generator(startf=16, maxf=512, layer_count=int(math.log(img_size,2)-1), latent_size=512, channels=3)
#     Gs.load_state_dict(torch.load(model_path+'Gs_dict.pth'))

#     Gm = Mapping(num_layers=int(math.log(img_size,2)-1)*2, mapping_layers=8, latent_size=512, dlatent_size=512, mapping_fmaps=512) #num_layers: 14->256 / 16->512 / 18->1024
#     Gm.load_state_dict(torch.load(model_path+'Gm_dict.pth'))

#     Gm.buffer1 = torch.load(model_path+'center_tensor.pt')
#     layer_num = int(math.log(img_size,2)-1)*2 # 14->256 / 16 -> 512  / 18->1024
#     layer_idx = torch.arange(layer_num)[np.newaxis, :, np.newaxis] # shape:[1,18,1], layer_idx = [0,1,2,3,4,5,6。。。，17]
#     ones = torch.ones(layer_idx.shape, dtype=torch.float32) # shape:[1,18,1], ones = [1,1,1,1,1,1,1,1]
#     coefs = torch.where(layer_idx < layer_num//2, 0.7 * ones, ones) # 18个变量前8个裁剪比例truncation_psi [0.7,0.7,...,1,1,1]

#     Gm.eval()
#     Gs.to(device)

# elif model_type == 'PGGAN':
#     model_path = '../checkpoint/pggan/pggan_celebahq1024.pth'#'../MTV-TSA/checkpoint/pggan/pggan_celebahq1024.pth'
#     generator = model_pggan.PGGANGenerator(resolution=img_size).to(device)
#     checkpoint = torch.load(model_path) #map_location='cpu'
#     if 'generator_smooth' in checkpoint: #默认是这个
#         generator.load_state_dict(checkpoint['generator_smooth'])
#     else:
#         generator.load_state_dict(checkpoint['generator'])
#     const1 = torch.tensor(0)

# else:
#     print('error in loading model')

# z_image = torch.zeros(id_e-id_s,512)
# w_image = torch.zeros(id_e-id_s,18,512)

# for i in range(id_e-id_s):
#     set_seed(i+id_s)
#     z = torch.randn(1,512)
#     if model_type == 'StyleGANv1':
#         with torch.no_grad():
#             w = Gm(z,coefs_m=coefs).to(device) #[batch_size,18,512]
#             w_image[i] = w[0]
#     z_image[i] = z[0]
#     print(i+id_s)

# torch.save(z_image.clone(),'./%s_z_%d_%d.pt'%(model_type,id_s,id_e))
# if model_type == 'StyleGANv1':
#     torch.save(w_image.clone(),'./%s_w_%d_%d.pt'%(model_type,id_s,id_e))
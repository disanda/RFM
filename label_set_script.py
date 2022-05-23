# made label_set, Nvidia face classifer with synthesized faces (random seed) 
import os
import math
import pickle
import numpy as np
import torch
import torchvision
from training_utils import *
import multiprocessing

import dnnlib
from training import misc
import tensorflow as tf
import dnnlib.tflib as tflib

from model.stylegan1.net import Generator, Mapping
import model.pggan.pggan_generator as model_pggan #PGGAN
import model.stylegan2_generator as model_v2 #StyleGANv2

path_dict={
0 : './checkpoint/attribute/celebahq-classifier-00-male.pkl',
1 : './checkpoint/attribute/celebahq-classifier-01-smiling.pkl',
2 : './checkpoint/attribute/celebahq-classifier-02-attractive.pkl',
3 : './checkpoint/attribute/celebahq-classifier-03-wavy-hair.pkl',
4 : './checkpoint/attribute/celebahq-classifier-04-young.pkl',
5 : './checkpoint/attribute/celebahq-classifier-05-5-o-clock-shadow.pkl', #胡子
6 : './checkpoint/attribute/celebahq-classifier-06-arched-eyebrows.pkl',
7 : './checkpoint/attribute/celebahq-classifier-07-bags-under-eyes.pkl',
8 : './checkpoint/attribute/celebahq-classifier-08-bald.pkl',
9 : './checkpoint/attribute/celebahq-classifier-09-bangs.pkl',
10 : './checkpoint/attribute/celebahq-classifier-10-big-lips.pkl',
11 : './checkpoint/attribute/celebahq-classifier-11-big-nose.pkl',
12 : './checkpoint/attribute/celebahq-classifier-12-black-hair.pkl',
13 : './checkpoint/attribute/celebahq-classifier-13-blond-hair.pkl',
14 : './checkpoint/attribute/celebahq-classifier-14-blurry.pkl',
15 : './checkpoint/attribute/celebahq-classifier-15-brown-hair.pkl',
16 : './checkpoint/attribute/celebahq-classifier-16-bushy-eyebrows.pkl', #浓密
17 : './checkpoint/attribute/celebahq-classifier-17-chubby.pkl', #胖乎乎
18 : './checkpoint/attribute/celebahq-classifier-18-double-chin.pkl',
19 : './checkpoint/attribute/celebahq-classifier-19-eyeglasses.pkl',
20 : './checkpoint/attribute/celebahq-classifier-20-goatee.pkl',
21 : './checkpoint/attribute/celebahq-classifier-21-gray-hair.pkl',
22 : './checkpoint/attribute/celebahq-classifier-22-heavy-makeup.pkl',
23 : './checkpoint/attribute/celebahq-classifier-23-high-cheekbones.pkl',
24 : './checkpoint/attribute/celebahq-classifier-24-mouth-slightly-open.pkl',
25 : './checkpoint/attribute/celebahq-classifier-25-mustache.pkl',
26 : './checkpoint/attribute/celebahq-classifier-26-narrow-eyes.pkl',
27 : './checkpoint/attribute/celebahq-classifier-27-no-beard.pkl',
28 : './checkpoint/attribute/celebahq-classifier-28-oval-face.pkl',
29 : './checkpoint/attribute/celebahq-classifier-29-pale-skin.pkl',
30 : './checkpoint/attribute/celebahq-classifier-30-pointy-nose.pkl',
31 : './checkpoint/attribute/celebahq-classifier-31-receding-hairline.pkl',
32 : './checkpoint/attribute/celebahq-classifier-32-rosy-cheeks.pkl',
33 : './checkpoint/attribute/celebahq-classifier-33-sideburns.pkl',
34 : './checkpoint/attribute/celebahq-classifier-34-straight-hair.pkl',
35 : './checkpoint/attribute/celebahq-classifier-35-wearing-earrings.pkl',
36 : './checkpoint/attribute/celebahq-classifier-36-wearing-hat.pkl',
37 : './checkpoint/attribute/celebahq-classifier-37-wearing-lipstick.pkl',
38 : './checkpoint/attribute/celebahq-classifier-38-wearing-necklace.pkl',
39 : './checkpoint/attribute/celebahq-classifier-39-wearing-necktie.pkl',
} #these neural networks refer to https://github.com/NVlabs/stylegan2/blob/master/metrics/linear_separability.py

#loading stylegan model in pytorch
use_gpu = False
img_size = 1024
device = torch.device("cuda" if use_gpu else "cpu")
model_type = 'StyleGANv1' # StyleGANv1 / StyleGANv2 / PGGAN 

if model_type == 'StyleGANv1':
    model_path = '../MTV-TSA/checkpoint/stylegan_v1/ffhq1024/'
    Gs = Generator(startf=16, maxf=512, layer_count=int(math.log(img_size,2)-1), latent_size=512, channels=3)
    Gs.load_state_dict(torch.load(model_path+'Gs_dict.pth'))

    Gm = Mapping(num_layers=int(math.log(img_size,2)-1)*2, mapping_layers=8, latent_size=512, dlatent_size=512, mapping_fmaps=512) #num_layers: 14->256 / 16->512 / 18->1024
    Gm.load_state_dict(torch.load(model_path+'Gm_dict.pth'))

    Gm.buffer1 = torch.load(model_path+'./center_tensor.pt')
    layer_num = int(math.log(img_size,2)-1)*2 # 14->256 / 16 -> 512  / 18->1024
    layer_idx = torch.arange(layer_num)[np.newaxis, :, np.newaxis] # shape:[1,18,1], layer_idx = [0,1,2,3,4,5,6。。。，17]
    ones = torch.ones(layer_idx.shape, dtype=torch.float32) # shape:[1,18,1], ones = [1,1,1,1,1,1,1,1]
    coefs = torch.where(layer_idx < layer_num//2, 0.7 * ones, ones) # 18个变量前8个裁剪比例truncation_psi [0.7,0.7,...,1,1,1]

    Gm.eval()
    Gs.to(device)

elif model_type == 'StyleGANv2':  # StyleGAN2
    model_path = '../MTV-TSA/checkpoint/stylegan_v2/stylegan2_ffhq1024.pth'
    generator = model_v2.StyleGAN2Generator(resolution=img_size).to(device)
    checkpoint = torch.load(model_path, map_location='cpu') #map_location='cpu'
    if 'generator_smooth' in checkpoint: #default
        generator.load_state_dict(checkpoint['generator_smooth'])
    else:
        generator.load_state_dict(checkpoint['generator'])
    synthesis_kwargs = dict(trunc_psi=0.7,trunc_layers=8,randomize_noise=False)
    #Gs = generator.synthesis
    #Gs.cuda()
    #Gm = generator.mapping
    #truncation = generator.truncation
#     const_r = torch.randn(batch_size)
#     const1 = generator.synthesis.early_layer(const_r).detach().clone() #[n,512,4,4]

elif model_type == 'PGGAN':
    model_path = '../MTV-TSA/checkpoint/pggan/pggan_celebahq1024.pth'
    generator = model_pggan.PGGANGenerator(resolution=img_size).to(device)
    checkpoint = torch.load(model_path) #map_location='cpu'
    if 'generator_smooth' in checkpoint: #默认是这个
        generator.load_state_dict(checkpoint['generator_smooth'])
    else:
        generator.load_state_dict(checkpoint['generator'])
    const1 = torch.tensor(0)

else:
    print('error in loading model')

def classify(i,images,path):
    dnnlib.tflib.init_tf()
    for j in path_dict:
        classifier = misc.load_pkl(path_dict[j])
        logits = classifier.get_output_for(images, None) 
        #predictions = tf.nn.softmax(tf.concat([logits, -logits], axis=1))
        flag2 = torch.tensor(logits.eval())
        print(i,j,flag2)
        w_attribute[i-s][j] = flag2 #多线程并不能真正把w_attribute存入
    torch.save(w_attribute[i-s].clone(),path+'w_attribute_%d.pt'%i)
    #del Gs
    #del classifier
    #session_1.clear_session()

#resize_fn = torchvision.transforms.Resize(size=(256,256)) # 低版本无法使用,PIL才可以

if __name__ == "__main__":

    s = 65000   # start
    e = 66000   # end

    if model_type == 'StyleGANv1' or 'StyleGANv2':
        w_image = torch.zeros(e-s,18,512)
    elif model_type == 'PGGAN':
        w_image = torch.zeros(e-s,512)
    else:
        print('model_type error')
        
    w_attribute = torch.zeros(e-s,40)

    resultPath = './label_set/'+model_type+'/' #./result/StyleGANv1/styleganv1_pytorchSeed(Img)_0-5000
    if not os.path.exists(resultPath): os.mkdir(resultPath)

    resultPath1 = resultPath+model_type + '_s%d_e%d_images/'%(s,e) # imgs
    if not os.path.exists(resultPath1): os.mkdir(resultPath1)

    resultPath2 = resultPath+model_type + '_s%d_e%d_w(z)/'%(s,e) # latent vector
    if not os.path.exists(resultPath2): os.mkdir(resultPath2)

    resultPath3 = resultPath+model_type + '_s%d_e%d_attri_svm_-1_1_dgx2/'%(s,e) # attribute vector
    if not os.path.exists(resultPath3): os.mkdir(resultPath3)

    for i in range(s,e):
        set_seed(i)
        z = torch.randn(1,512)
        if model_type == 'StyleGANv1':
            with torch.no_grad():
                w = Gm(z,coefs_m=coefs).to(device) #[batch_size,18,512]
            imgs1 = Gs.forward(w,int(math.log(img_size,2)-2))

        elif model_type == 'StyleGANv2':
            with torch.no_grad():
                #use generator
                result_all = generator(z.to(device), **synthesis_kwargs)
                imgs1 = result_all['image']
                w = result_all['wp']

        elif model_type == 'PGGAN':
            w = z.to(device)
            with torch.no_grad(): #这里需要生成图片和变量
                result_all = generator(w)
            imgs1 = result_all['image']
        imgs1 = torch.nn.functional.interpolate(imgs1, size=(256,256))
        w_image[i-s] = w[0]
        #以下两端保存照片和w，可以取消减少空间
        #torch.save(w_image[i-s].clone(),resultPath2+'w_%d.pt'%(i))
        #torchvision.utils.save_image(imgs1*0.5+0.5, resultPath1+'%s_%s_256.jpg'%(str(i).rjust(5,'0'),model_type))
        images = imgs1.detach().clone().numpy()
        p = multiprocessing.Process(target=classify,args=(i,images,resultPath3)) # 判断属性
        p.start()
        p.join()

    #以下多线程未存入
    #torch.save(w_image.clone(),'./%s_w_%d_%d.pt'%(model_type,s,e)) 
    #torch.save(w_attribute.clone(),'./%s_w_attribute_%d_%d.pt'%(model_type,s,e))

# styleganv1 官方分类器 分类属性（ 用svm 分类到 [-a, a], 未进过softmax得到概率[0,1] )
# tf-1.14 session 无法释放内存, 使用多线程解决
# 这个版本使用pytorch的Gm来生成图片 pytorch=1.5.1, torchvisiion=0.6.1 :handle lerp() and transform(in torchvision)

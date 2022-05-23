# -*- coding: utf-8 -*-
# Merge all single attribute files into one file (torch tensor format), n * (1,40) => (n,40)， n default as 1000
# 分类器输出的是单个w的属性，将他们合并到一起:  n * (1,40) => (n,40),  n 默认为 1000

import torch
mtype = 'StyleGANv2' # pggan or styleganv1
#w_type = 0 # 0: attribute vector(svm), 1: latent space (pggan: [n, 512] / stylega: [n, 18, 512])
d = 65000
id_s = d
id_e = d+1000
dgx = 'dgx2'

#path = './result/StyleGANv1/StyleGANv1_s%d_e%d_attri(svm[-1,1])/w_attribute_'%(id_s,id_e)
#path = './result/StyleGANv1/StyleGANv1_s3755_e3760_attri(svm[-1,1])/w_attribute_' #'./stylegan/wAttribute_0-5000/w_attribute_4078.pt'
#path = './result/PGGAN/PGGAN_s%d_e%d_attri(svm[-1,1])/w_attribute_'%(id_s,id_e)
path = './result/StyleGANv2/StyleGANv2_s%d_e%d_attri_svm_-1_1_%s/w_attribute_'%(id_s,id_e,dgx)

tensor_attribute =  torch.zeros(id_e-id_s,40)
#tensor_attribute =  torch.zeros(id_e-id_s,18,512)

n = 0
for i in range(id_e-id_s):
    try:
        flag = torch.load(path+str(i+id_s)+'.pt')
        tensor_attribute[i] = flag # torch.any(flag) ,全0为False
    except:
        print('failed load %d'%(i+id_s))
        n = n + 1 
    #else:
        #print(i, flag.shape)
        #print(flag)

print('finish_failed:%d'%n)

torch.save(tensor_attribute.clone(),'./%s_attribute_seed%d_%d.pt'%(mtype,id_s,id_e))

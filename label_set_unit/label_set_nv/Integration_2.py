# Combine label files: m * (n,40) => (n*m,40), m default as 5000
#将多个属性集合并为一个更大的集合文件:  m * (n,40) => (n*m,40)

import torch
mtype = 'StyleGANv2' # pggan or styleganv1
#w_type = 0 # 0: attribute vector(svm), 1: latent space (pggan: [n, 512] / stylega: [n, 18, 512])
id_s = 55000
id_e = 65000

path = './%s_attribute_seed'%mtype
#path = './result/StyleGANv1/StyleGANv1_s3755_e3760_attri(svm[-1,1])/w_attribute_' #'./stylegan/wAttribute_0-5000/w_attribute_4078.pt'

tensor_attribute =  torch.zeros(id_e-id_s,40)
#tensor_attribute =  torch.zeros(id_e-id_s,18,512)

for i in range( (id_e-id_s)//1000 ) :
    try:
        flag = torch.load(path+str(id_s+i*1000)+'_'+str(id_s+(i+1)*1000)+'.pt')
    except:
        print('failed load %d'%(i+id_s))
    #else:
        #print(i, flag.shape)
        #print(flag)
        
    tensor_attribute[i*1000:(i+1)*1000] = flag # torch.any(flag) ,全0为False

print('finish')

torch.save(tensor_attribute.clone(),'./%s_attribute_seed%d_%d.pt'%(mtype,id_s,id_e))
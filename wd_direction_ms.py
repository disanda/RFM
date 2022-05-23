import os
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

import torch
import pickle
import gzip
import math
import torchvision
import matplotlib.pylab as plt
import sys
sys.path.append('../')
from model.stylegan1.net import Generator, Mapping #StyleGANv1

#dict1 labels: [0,1]
with open('./checkpoint/stylegan_v1/latent_training_data.pkl.gz', 'rb') as f:
   z, w, labels = pickle.load(gzip.GzipFile(fileobj=f)) # (20_307,512), (20307, 18, 512), dict
w_attribute40 = torch.load('./checkpoint/stylegan_v1/styleganv1_20307_attribute40_oneHot_probability.pt') # [20307,40]
print(w.shape)
layers = 0
layere = 18
w = w.reshape(w.shape[0],(layere-layers)*512)
w = torch.tensor(w[:,layers:layere,:])

## interpretable directions, cancel code annotations 
# index = 400*3
# print(attri_d[index:index+400])
# print(np.sum(attri_d>0.3)) #测试border范围的个数

attri_id = 24
attri_d = w_attribute40[:,attri_id].numpy() 
# 0_smlie 11_noGlasses, 12_readingGlasses, 13_sunglasses, 14_anger, 15_contempt, 16_disgust, 17_fear, 18_happiness, 19_neutral
# 20_sadness, 21_surprise, 30_hair_Bald, 38_SwimmingGoggles,

attri_d[attri_d >= 0.5] = 1 
attri_d[attri_d < 0.5 ] = 0

## other specical cases
# attri_d[attri_d > 0.] = 1    # 22_overExposure, layer:6-9, 23_eyeMakeup, 24_lip_makeup
# attri_d[attri_d == 0 ] = 0

# attri_d[attri_d > 0.7] = 1   # 34_hair_blond, acc:0.98 layer:0-9 male or 4-12 female
# attri_d[attri_d <= 0.7 ] = 0

# attri_d[attri_d >= 0.8] = 1  # 35_hair_black, acc:0.8874 layer:
# attri_d[attri_d < 0.8 ] = 0

# attri_d[attri_d > 0.0] = 1   # 36_hair_red, acc:0.9918 layer: 4-12 clip-0.02
# attri_d[attri_d == 0.0 ] = 0

# attibute classifer 属性分类器
penalty_l = 'l2' # l1, l2
solver = 'saga' # 'sag', 'newton-cg', 'lbfgs', 'liblinear', 'saga'  
max_iter = 500
#multi_class='multinomial',这个参数会让属性变化不敏感，即d*10才能达到相同的效果
clf = LogisticRegression(penalty= penalty_l,  solver=solver, verbose=2, max_iter=max_iter)  

# clf = LogisticRegression(penalty= 'l1', dual= False, tol=1e-4, C = 1.0, fit_intercept=True,\
#    intercept_scaling=1, class_weight={1,20}, random_state=None, solver='saga',\
#    max_iter=100, multi_class='auto', verbose=0,  warm_start = False,)

# from sklearn.svm import SVC
# penalty_l = 'SVC'
# solver = 'linear'
# max_iter = -1
# clf = SVC(kernel=solver, decision_function_shape='ovr',verbose=2) # ovo/ovr

## SGD classifer
# loss = 'hinge'
# penalty = 'l2'
# class_weight_f = None #'balanced' None
# iteration= 500
# penalty_l = loss+'-'+penalty+'-'+str(class_weight_f)+'-'+str(iteration)  #+ class_weight
# clf = SGDClassifier(loss=loss, penalty=penalty, class_weight=class_weight_f, early_stopping=False, verbose=2, max_iter=iteration, tol= 0.000001)
# loss: 'hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 
# 'squared_error', 'huber', 'epsilon_intensitive', 'squared_epsilon_intensitive'
# penalty: 'l1', 'l2'(default) , 'elasticnet'
# class_weight: None, 'balanced', {x,y}

# from sklearn.gaussian_process import GaussianProcessClassifier
# clf = GaussianProcessClassifier()

# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier()

#from sklearn.svm import LinearSVC
#penalty_l = 'LinearSVC'
#clf = LinearSVC(penalty = 'l2', verbose = 2) # default: l2, dual-True (flase when n_samples>n_features), classWeight-None, multi_class: ovr/crammer_singer

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# penalty_l = 'LDA-svd' 
# clf = LinearDiscriminantAnalysis(solver="svd", store_covariance=True) # svd, lsqr

# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# penalty_l = 'QDA'
# clf = QuadraticDiscriminantAnalysis()

#-------------------------训练数据集，并测试准确性--------------------------------------
samples = 8000
clf.fit(w[:samples], attri_d[:samples])
print(clf.coef_.shape)
attri_direction = clf.coef_#.reshape((latere-layers, 512))

# accuracy = cross_val_score(clf, w, attri_d, scoring='accuracy', cv=5)
attri_d_predict = clf.predict(w)
accuracy = accuracy_score(attri_d,attri_d_predict)
print(accuracy)

np.save('./id%d_dict1_%s_%dk_acc%.5f_%s_iter%d.npy'%(attri_id,penalty_l,samples//1000,accuracy,solver,max_iter),clf.coef_) # layere-layers

## Real image via StyleGANv1 pre-trained Params
#name = 'id12-i0-w1400-norm351.482544.pt' # i1_dy.pt, i2_zy.pt / id6-i0-w1500-norm319.548279.pt, id6-i0-w1400-norm302.049072.pt , i3_cxx2_norm730, 06-norm4009.pt / i4_msk, 12 / i5_ty, 13/ i15 bald_man / i16 woman_1 /i17_xlz.pt / 18_man1 / 19_man2 / 20_man3 / 21 woman_2
#name = 'id12-i0-w2000-norm98.228111-imgLoss1.730433.pt'
#name = 'id12-iter1541-norm768.978210-imgLoss-min0.965121.pt'
name = 'id12-3066.pt'
use_gpu = False
device = torch.device("cuda" if use_gpu else "cpu")
img_size = 1024
GAN_path = './checkpoint/stylegan_v1/ffhq1024/'
#direction = 'eyeglasses' #smile, eyeglasses, pose, age, gender
#direction_path = './latentvectors/directions/stylegan_ffhq_%s_w_boundary.npy'%direction
w_path = './checkpoint/styleGAN_v1_w/face_w/%s'%name

# #discovering face semantic attribute dirrections 
#attri_id = 0
id_path_1 = 'id3_dict1_l1_8k_acc0.88708_saga_iter2000_0.5_1092.npy'
id_path = './attribute_vec/dict1_vec/direction/id3_Yaw/' + id_path_1
#id_path = './result/attribute_vector_img/dict1_vec/id14_anger_direction.npy' 
direction1 = torch.tensor(np.load(id_path))#.reshape(2,18,512)[1] #_l1_iter300_acc0991
#direction2 = torch.tensor(np.load(id_path_2))
#print(direction1.reshape(-1)@direction2.reshape(-1))
#direction = direction.reshape(1,1,512).repeat(1,18,1) # z -> w

#Loading Pre-trained Model, Directions
Gs = Generator(startf=16, maxf=512, layer_count=int(math.log(img_size,2)-1), latent_size=512, channels=3)
Gs.load_state_dict(torch.load(GAN_path+'Gs_dict.pth', map_location=device))

w = torch.load(w_path, map_location=device).clone().squeeze(0) # face
#w[w!=0.0]=0.0
print(w.norm())
# # x = w.reshape(-1)
# # print(x[x>100])
# # # # w = w / w.norm()
# # # # print(w.norm())

##减去接近0的值(0.01-0.03),人更不容易变化
# flag = w.detach().numpy()
# plt.hist(flag.reshape(-1), bins=512, color='blue', alpha=0.5, label='flag')
# plt.legend()
# plt.show()

layers = 0
layere = 18
direction1 = direction1.view(layere-layers,512)
clip1 = 0.0 # 0.01-0.03
direction1[torch.abs(direction1)<=clip1] = 0.0

bonus1= -100 #bonus   (-10) <- (-5) <- 0 ->5 ->10
start1= 0 # default 0, if not 0, will be bed performance
end1=   3  # default 3 or 4. if 3, it will keep face features (glasses). if 4, it will keep dirrection features (Smile).
w[start1:end1] = (w+bonus1*direction1)[start1:end1] #w = w + bonus*direction all_w


## 再加一次向量 (因为同属性的向量也是垂直的)
# direction2 = direction2.view(layere-layers,512)
# clip2 = 0.00 # 0.01-0.03
# direction2[torch.abs(direction2)<=clip2] = 0.0

# bonus2= 40 #bonus   (-10) <- (-5) <- 0 ->5 ->10
# start2= 6 # default 0, if not 0, will be bed performance
# end2=   13  # default 3 or 4. if 3, it will keep face features (glasses). if 4, it will keep dirrection features (Smile).
# w[start2:end2] = (w+bonus2*direction2)[start2:end2] #w = w + bonus*direction all_w


w = w.reshape(1,18,512)
with torch.no_grad():
  img = Gs.forward(w,8) # 8->1024

torchvision.utils.save_image(img*0.5+0.5, \
   './img_%s__bonus1_%.2f_start%d_end%d_face%s_clip%s.png'\
   %(id_path_1[:-4],bonus1,start1,end1,name,clip1))

# torchvision.utils.save_image(img*0.5+0.5, \
#    './img(%s)_bonus1-%.2f_bonus2-%.2f_start%d_end%d_face%s_aId%d_iter%d_clip%s_s%d_e%d.png'\
#    %(id_path_1[7:-4]+id_path_2[7:-4],bonus1,bonus2,start1,end1,name,attri_id,100,clip1,layers,layere))







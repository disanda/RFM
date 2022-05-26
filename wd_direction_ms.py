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
with open('./checkpoint/label_dict/stylegan1/ms/latent_training_data.pkl.gz', 'rb') as f:
   z, w, _ = pickle.load(gzip.GzipFile(fileobj=f)) # (20_307,512), (20307, 18, 512), dict
w_attribute40 = torch.load('./checkpoint/label_dict/stylegan1/ms/stylegan1_20307_attributes40_ms.pt') # [20307,40]
print(w.shape)
layers = 0
layere = 18
w = torch.tensor(w[:,layers:layere,:])
w = w.reshape(w.shape[0],(layere-layers)*512)


## interpretable directions
attri_id = 7
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
penalty_l = 'l1' # l1, l2
solver = 'saga' # 'sag', 'newton-cg', 'lbfgs', 'liblinear', 'saga'  
max_iter = 500
#multi_class='multinomial',这个参数会让属性变化不敏感，即d*10才能达到相同的效果
clf = LogisticRegression(penalty= penalty_l,  solver=solver, verbose=2, max_iter=max_iter)  
## clf = LogisticRegression(penalty= 'l1', dual= False, tol=1e-4, C = 1.0, fit_intercept=True,\
##    intercept_scaling=1, class_weight={1,20}, random_state=None, solver='saga',\
##    max_iter=100, multi_class='auto', verbose=0,  warm_start = False,)

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
samples = 8000 #12000
clf.fit(w[:samples], attri_d[:samples])
print(clf.coef_.shape)
attri_direction = clf.coef_#.reshape((latere-layers, 512))

# accuracy = cross_val_score(clf, w, attri_d, scoring='accuracy', cv=5)
attri_d_predict = clf.predict(w)
accuracy = accuracy_score(attri_d,attri_d_predict)
print(accuracy)

np.save('./age_id%d_dict1_%s_%dk_acc%.5f_%s_iter%d.npy'%(attri_id,penalty_l,samples//1000,accuracy,solver,max_iter),clf.coef_) # layere-layers


# torchvision.utils.save_image(img*0.5+0.5, \
#    './img(%s)_bonus1-%.2f_bonus2-%.2f_start%d_end%d_face%s_aId%d_iter%d_clip%s_s%d_e%d.png'\
#    %(id_path_1[7:-4]+id_path_2[7:-4],bonus1,bonus2,start1,end1,name,attri_id,100,clip1,layers,layere))







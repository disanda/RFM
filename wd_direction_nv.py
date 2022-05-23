# 矩阵排序， 同一个属性最高前n个 ， 减去同一个属性最低前n个
# 同一个人，表情不同的潜向量相减 == 表情潜向量 ？ 1.相减后负值清零(亦或微小值清零) 2.属性在合理的地方起步:(胡子从男性开始) 
# 不同层代表不同向量
# 数值分析+概率分布 or 空间分析+线性代数

# 分类器准确性越高，属性效果越好

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
from training_utils import *

## dict2 labels: [-1,1]

w_attribute40 = torch.load('./checkpoint/stylegan_v1/dict2/styleganv1_attribute_seed0_30000') # [n,40]
w = torch.load('./checkpoint/stylegan_v1/dict2/StyleGANv1_w_0_30000.pt') #[n,18,512]
#w = w.view(8000,18*512)

# attri_id = 4
# attri_d = w_attribute40[:,attri_id].numpy() # id,  acc=,  layer: 
# print(attri_d[:100])
# attri_d[attri_d >= 0.0] = 1.0 #attri_d[attri_d == 0.0 ] = 0.0
# attri_d[attri_d < 0.0]  = -1.0
# print(sum(attri_d==-1.0)) #统计个数

# ## 属性分类器
# clf = LogisticRegression(penalty= 'l1', dual= False, tol=1e-4, C = 1.0, fit_intercept=True,\
#    intercept_scaling=1, class_weight={1,20}, random_state=None, solver='saga',\
#    max_iter=100, multi_class='auto', verbose=0,  warm_start = False,)

# penalty_l = 'l1' # l1, l2
# solver = 'saga' # 'sag', 'newton-cg', 'lbfgs', 'liblinear', 'saga'
# max_iter = 2000
# clf = LogisticRegression(penalty= penalty_l,  solver=solver, verbose=2, max_iter=max_iter)  
# multi_class='multinomial',这个参数会让属性变化不敏感，即d*10才能达到相同的效果

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

# from sklearn.svm import LinearSVC
# penalty_l = 'SVC_crammer_singer'
# clf = LinearSVC() # default: l2, dual-True (flase when n_samples>n_features), classWeight-None, multi_class: ovr/crammer_singer

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# penalty_l = 'LDA-svd' 
# clf = LinearDiscriminantAnalysis(solver="svd", store_covariance=True) # svd, lsqr

# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# penalty_l = 'QDA'
# clf = QuadraticDiscriminantAnalysis()

#-------------------------训练数据集，并测试准确性--------------------------------------
# with open('./dict1_id%d_model%s.txt'%(attri_id,penalty_l), 'a+') as f:
#    print('neg_samples:%d'%sum(attri_d==-1.0),file=f)

# for i in range(12):
#    samples = 1000 + 1000*i
#    from datetime import datetime, time
#    time1 = datetime.now() 
#    x = clf.fit(w[:samples], attri_d[:samples])
#    time2 = datetime.now()
#    print('fit time:%f seconds'%(time2-time1).seconds)
#    print(clf.coef_.shape)
#    attri_direction = clf.coef_#.reshape((latere-layers, 512))

#    # accuracy = cross_val_score(clf, w, attri_d, scoring='accuracy', cv=5)
#    attri_d_predict = clf.predict(w)
#    accuracy = accuracy_score(attri_d,attri_d_predict)
#    print(accuracy)
#    np.save('./id%d_dict1_%s_%dk_acc%.5f_%s_iter%d_0.5_1092.npy'%(attri_id,penalty_l,samples//1000,accuracy,solver,max_iter),clf.coef_) # layere-layers

#    with open('./dict1_id%d_model%s.txt'%(attri_id,penalty_l), 'a+') as f:
#        print('samples:%d'%samples,file=f)
#        print('fit time:%f seconds'%(time2-time1).seconds,file=f)
#        print(accuracy,file=f)


# Cleaning face attribute labels file (latent_training_data.pkl). 
# It size is (20307,40) that denotes 20307 samples with 40 attributes (w,d) 
# We change all attribute labels into a unified one hot format (0-1)
# 清洗人脸属性标签数据集(latent_training_data.pkl),我们将其中的属性全部人脸属性标签改为统一的one-hot格式(0-1)

import gzip
import math
import pickle
import torch
import numpy as np
import torchvision
from model.stylegan1.net import Generator, Mapping

import training_utils
training_utils.set_seed(0)

with open('./checkpoint/stylegan_v1/latent_training_data.pkl.gz', 'rb') as f:
   z, w, labels = pickle.load(gzip.GzipFile(fileobj=f)) #labels :20307, dict_keys(['faceId', 'faceRectangle', 'faceLandmarks', 'faceAttributes']

# test first 300 samples:
# print(labels[0]['faceAttributes'], len(labels))
# for i,j in enumerate(labels[:300]):
#    if j['faceAttributes']['hair']['hairColor'] != []:
#     print(i,j['faceAttributes']['hair']['hairColor'][0])
#    else:
#     print(i)

#one-hot
flag=0
attribute1 = np.zeros((20307,40))
attribute1 = torch.from_numpy(np.float32(attribute1)) # np.float64 -> torch.float32
for i,j in enumerate(labels):
    attribute1[i][0]  = j['faceAttributes']['smile'] # 00_smile

    if j['faceAttributes']['headPose']['roll'] >= 0:
        attribute1[i][1] = j['faceAttributes']['headPose']['roll'] # 01_head_Pose_Roll
        attribute1[i][2] = 0
    else:
        attribute1[i][1] = 0 # 01_head_Pose_Roll
        attribute1[i][2] = j['faceAttributes']['headPose']['roll'] * (-1.0) # 02_head_Pose_Roll_Neg

    if j['faceAttributes']['headPose']['yaw'] >= 0:
        attribute1[i][3] = j['faceAttributes']['headPose']['yaw'] # 03_head_Pose_Yaw
        attribute1[i][4] = 0
    else:
        attribute1[i][3] = 0 
        attribute1[i][4] = j['faceAttributes']['headPose']['yaw'] * (-1.0) # 04_head_Pose_Yaw_Neg

    if j['faceAttributes']['gender'] == 'male':
        attribute1[i][5]  = 1.0 # 05 gender_male
        attribute1[i][6]  = 0   # 06 gender_female
    elif j['faceAttributes']['gender'] == 'female':
        attribute1[i][5]  = 0 
        attribute1[i][6]  = 1.0
    else:
        print('error in 5-6')
        break

    attribute1[i][7]  = j['faceAttributes']['age'] # 07_age
    attribute1[i][8]  = j['faceAttributes']['facialHair']['moustache'] # 08_moustache
    attribute1[i][9]  = j['faceAttributes']['facialHair']['beard'] # 09_beard
    attribute1[i][10]  = j['faceAttributes']['facialHair']['sideburns'] # 10_sideburns

    if j['faceAttributes']['glasses'] == 'NoGlasses': 
        attribute1[i][11] = 1.0 # 11_noGlasses
        attribute1[i][12] = 0   # 12_ReadingGlasses
        attribute1[i][13] = 0   # 13_Sunglasses
        attribute1[i][38] = 0
    elif j['faceAttributes']['glasses'] == 'ReadingGlasses': 
        attribute1[i][11] = 0
        attribute1[i][12] = 1.0
        attribute1[i][13] = 0
        attribute1[i][38] = 0
    elif j['faceAttributes']['glasses'] == 'Sunglasses':
        attribute1[i][11] = 0
        attribute1[i][12] = 0
        attribute1[i][13] = 1.0
        attribute1[i][38] = 0
        #print('Sunglasses')
    elif j['faceAttributes']['glasses'] == 'SwimmingGoggles':
        attribute1[i][11] = 0
        attribute1[i][12] = 0
        attribute1[i][13] = 0
        attribute1[i][38] = 1.0 # SwimmingGoggles
    else:
        print(j['faceAttributes']['glasses'])
        print('error in 11-13')
        break

    attribute1[i][14] = j['faceAttributes']['emotion']['anger'] # 14_anger
    attribute1[i][15] = j['faceAttributes']['emotion']['contempt'] # 15_contempt
    attribute1[i][16] = j['faceAttributes']['emotion']['disgust'] # 16_disgust
    attribute1[i][17] = j['faceAttributes']['emotion']['fear'] # 17_fear
    attribute1[i][18] = j['faceAttributes']['emotion']['happiness'] # 18_happiness
    attribute1[i][19] = j['faceAttributes']['emotion']['neutral'] # 19_neutral
    attribute1[i][20] = j['faceAttributes']['emotion']['sadness'] # 20_sadness
    attribute1[i][21] = j['faceAttributes']['emotion']['surprise'] # 21_surprise

    if j['faceAttributes']['exposure']['exposureLevel'] == 'overExposure':
        attribute1[i][22] = j['faceAttributes']['exposure']['value'] # 22_overExposure
    else:
        attribute1[i][22] = 0

    if j['faceAttributes']['makeup']['eyeMakeup'] == True:
        attribute1[i][23] = 1.0 # 23_eyeMakeup
    elif j['faceAttributes']['makeup']['eyeMakeup'] == False:
        attribute1[i][23] = 0
    else:
        print('error in 23')
        break

    if j['faceAttributes']['makeup']['lipMakeup'] == True:
        attribute1[i][24] = 1.0 # 24_lipMakeup
    elif j['faceAttributes']['makeup']['lipMakeup'] == False:
        attribute1[i][24] = 0
    else:
        print('error in 24')
        break

    if j['faceAttributes']['accessories'] == []:
        attribute1[i][25] = 0 # 25_headwear
        attribute1[i][26] = 0 # 26_glasses
        attribute1[i][39] = 0 # 39_mask 模糊块
    elif j['faceAttributes']['accessories'][0]['type'] == 'glasses':
        attribute1[i][25] = 0
        attribute1[i][26] = j['faceAttributes']['accessories'][0]['confidence']
        attribute1[i][39] = 0 
    elif j['faceAttributes']['accessories'][0]['type'] == 'headwear':
        attribute1[i][25] = j['faceAttributes']['accessories'][0]['confidence']
        attribute1[i][26] = 0
        attribute1[i][39] = 0 
    elif j['faceAttributes']['accessories'][0]['type'] == 'mask':
        attribute1[i][25] = 0
        attribute1[i][26] = 0
        attribute1[i][39] = j['faceAttributes']['accessories'][0]['confidence']
    else:
        print('error in attribute25-26')
        print(j['faceAttributes']['accessories'])
        #attribute1[i][25] = 0 # 25_headwear
        #attribute1[i][26] = 0 # 26_glasses
        break
        
    if j['faceAttributes']['occlusion']['foreheadOccluded'] == False:
        attribute1[i][27] == 0
    elif j['faceAttributes']['occlusion']['foreheadOccluded'] == True:
        attribute1[i][27] == 1
    else:
        print('error in attribute24')
        break

    if j['faceAttributes']['occlusion']['eyeOccluded'] == False:
        attribute1[i][28] == 0
    elif j['faceAttributes']['occlusion']['eyeOccluded'] == True:
        attribute1[i][28] == 1
    else:
        print('error in attribute25')
        break

    if j['faceAttributes']['occlusion']['mouthOccluded'] == False:
        attribute1[i][29] == 0
    elif j['faceAttributes']['occlusion']['mouthOccluded'] == True:
        attribute1[i][29] == 1
    else:
        print('error in attribute26')
        break

    attribute1[i][30] = j['faceAttributes']['hair']['bald']

    if j['faceAttributes']['hair']['invisible'] == False:
        attribute1[i][31] = 0
    elif j['faceAttributes']['hair']['invisible'] == True:
        attribute1[i][31] = 1
    else:
        print('error in attribute28')
        break

    if j['faceAttributes']['hair']['hairColor'] == []:
        attribute1[i][32] = 0
        attribute1[i][33] = 0
        attribute1[i][34] = 0
        attribute1[i][35] = 0
        attribute1[i][36] = 0 
        attribute1[i][37] = 0
    elif j['faceAttributes']['hair']['hairColor'][0]['color'] == 'brown':
        attribute1[i][32] = j['faceAttributes']['hair']['hairColor'][0]['confidence'] # brown
        attribute1[i][33] = 0
        attribute1[i][34] = 0
        attribute1[i][35] = 0
        attribute1[i][36] = 0 
        attribute1[i][37] = 0
    elif j['faceAttributes']['hair']['hairColor'][0]['color'] == 'gray':
        attribute1[i][32] = 0
        attribute1[i][33] = j['faceAttributes']['hair']['hairColor'][0]['confidence'] # gray
        attribute1[i][34] = 0
        attribute1[i][35] = 0
        attribute1[i][36] = 0 
        attribute1[i][37] = 0
    elif j['faceAttributes']['hair']['hairColor'][0]['color'] == 'blond':
        attribute1[i][32] = 0
        attribute1[i][33] = 0
        attribute1[i][34] = j['faceAttributes']['hair']['hairColor'][0]['confidence'] # blond
        attribute1[i][35] = 0
        attribute1[i][36] = 0 
        attribute1[i][37] = 0
    elif j['faceAttributes']['hair']['hairColor'][0]['color'] == 'black':
        attribute1[i][32] = 0
        attribute1[i][33] = 0
        attribute1[i][34] = 0
        attribute1[i][35] = j['faceAttributes']['hair']['hairColor'][0]['confidence'] # black
        attribute1[i][36] = 0 
        attribute1[i][37] = 0
    elif j['faceAttributes']['hair']['hairColor'][0]['color'] == 'red':
        attribute1[i][32] = 0
        attribute1[i][33] = 0
        attribute1[i][34] = 0
        attribute1[i][35] = 0
        attribute1[i][36] = j['faceAttributes']['hair']['hairColor'][0]['confidence'] # red
        attribute1[i][37] = 0
    elif j['faceAttributes']['hair']['hairColor'][0]['color'] == 'other':
        attribute1[i][32] = 0
        attribute1[i][33] = 0
        attribute1[i][34] = 0
        attribute1[i][35] = 0
        attribute1[i][36] = 0 
        attribute1[i][37] = j['faceAttributes']['hair']['hairColor'][0]['confidence'] # other
    else:
        print('error in attribute29-34')
        break
    print('finish_%d'%i)

torch.save(attribute1, './styleganv1_20307_attribute40_oneHot_probability.pt')

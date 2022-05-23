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

# print(labels[0]['faceAttributes'], len(labels))
# for i,j in enumerate(labels[:300]):
#    if j['faceAttributes']['hair']['hairColor'] != []:
#     print(i,j['faceAttributes']['hair']['hairColor'][0])
#    else:
#     print(i)

# 将 dict1 中labels 归一化到一个np数组中, one-hot with probability
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


# labels_dict1={ #[0->1]
#     '00_smile':0.1, 
#     '01_head_Pose_Roll': 0-10,
#     '02_head_Pose_Roll_Neg': 0-10,
#     '03_head_Pose_Yaw': 0-10,
#     '04_head_Pose_Yaw_Neg': 0-10,
#     '05_gender_male':0 or 1, # Male or Female
#     '06_gender_female':0 or 1, # Male or Female
#     '07_age':0-70,
#     '08_moustache':0.001, #小胡子
#     '09_beard':0.001,
#     '10_sideburns':0.001,
#     '11_noGlasses': 0 or 1, (1, 16396)
#     '12_readingGlasses': 0 or 1, (1, 32)
#     '13_sunglasses': 0 or 1, (1, 903个)
#     '14_anger': 0-1, 样本数量太少, 考虑用nn来做 (0.5,14个; 0.3,32个) acc=0.9993(0.5), 0.998473(0.3) #效果不明显
#     '15_contempt': 0-1, 样本数量太少 (0.3,9个; 0.5,1个)
#     '16_disgust': 0-1, 样本数量太少 (0个)
#     '17_fear': 0-1, 样本数量太少（0.5,2个; 0.3,3个）
#     '18_happiness': 0-1, 大量样本集中在此
#     '19_neutral': 0-1, 大量样本集中在此
#     '20_sadness': 0-1, border=0.5, 54个样本
#     '21_surprise': 0.001,
#     '22_overExposure': 0 or 0.7-1,
#     '23_eyeMakeup': 0 or 1,
#     '24_lipMakeup': 0 or 1,
#     '25_headwear': 0.001,
#     '26_glasses': 0.001,
#     '27_occluded_Forehead': 0.001,
#     '28_occluded_Eye': 0.001,
#     '29_occluded_Mouth': 0.001,
#     '30_hair_Bald': 0-1 (0.5)
#     '31_hair_Invisible': 0.001,
#     '32_hairColor_Brown:': 0.001,   #棕色
#     '33_hairColor_Gray:': 0.001,    #灰色
#     '34_hairColor_Blond': 0-1, <0.7 #金色 5-11 (4-12)
#     '35_hairColor_Black': 0-1,      #黑色
#     '36_hairColor_Red': 0 or around 1 #红色
#     '37_hairColor_other': 0.001,
#     'attribute1[i][38] = 1.0 # SwimmingGoggles 29个
#     'attribute1[i][39] = 1.0 # accessories-mask
# }

# labels_dict2={ #celebahq-classifier , [-1->1]
#     '00-male': [-1,1], # 10k:[5713-4287], 8k:[4582-3218] -1 男， 1 女
#     '01-smiling': [-1,1], # -1 不笑 1 笑 , layer:2-3
#     '02-attractive': . 
#     '03-wavy-hair': . # layer:3-4:女性头发增减. +减少(变直)头发 -增加头发. SVC_ovo(能用且ovr一样), LR_l2:能用，敏感. LR_l1:能用，且解耦充分
#     '04-young': .
#     '05-5-o-clock-shadow': [-1,1],
#     '06-arched-eyebrows': # (6013-1987) 弯眉毛, -眉毛往上 +眉毛往下 layer:3-4(that points only 3), 三个分类器都能用，但是都和笑耦合
#     '07-bags-under-eyes': [-1,1],
#     '08-bald': # (7934-66) (9924-76) SVC(最好)，比较慢，但比较准(准确率100).  LR_L2快, LR_L1慢但稀疏, 但两者都不好.
#     '09-bangs': #刘海 () (8538-1462)
#     '10-big-lips': 0.001,
#     '11-big-nose': 0.001, 
#     '12-black-hair': 0.001,
#     '13-blond-hair': 0.001,
#     '14-blurry': 0.001,
#     '15-brown-hair': 0.001,
#     '16-bushy-eyebrows': #浓密的眉毛 6604-1396 layer2-3耦合性别，3-4耦合微笑(嘴巴) 4-5相对明显
#       18080-3920 
#     '17-chubby': 0.001, #胖乎乎的
#     '18-double-chin': 0.001,
#     '19-eyeglasses': 0.001,
#     '20-goatee': #山羊胡子.  layer:3-4. -增加胡子(嘴巴周围). saga_l1不改变嘴型, l2改变嘴型,  
#     '21-gray-hair': 0.001,
#     '22-heavy-makeup': #浓妆 6933-1067, layer:4-5最佳，SVC最佳(小样本最佳？)
#     '23-high-cheekbones': #高颚骨
#     '24-mouth-slightly-open': 0.001,
#     '25-mustache': #胡子
#     '26-narrow-eyes': #狭窄的眼睛，没有样本
#     '27-no-beard': 0.001,
#     '28-oval-face': 0.001,
#     '29-pale-skin': #苍白的肤色
#     '30-pointy-nose': #尖鼻子
#     '31-receding-hairline':#后退的发际线
#     '32-rosy-cheeks': 0.001, #红润的脸颊, 失效，貌似样本太少？(7868-132)
#     '33-sideburns': 0.001, #鬓角 layer4-5最佳(3-4耦合嘴巴), 基本上就是胡子渣(带一点鬓角)
#     '34-straight-hair': #直发 id29 -头发变少 +头发变多 svc解耦更好
#     '35-wearing-earrings': #
#     '36-wearing-hat': # 7598-402
#     '37-wearing-lipstick': #
#     '38-wearing-necklace': #颈链
#     '39-wearing-necktie': #领带-领结
# }


'''
{'faceId': 'b6807d9a-0ab5-4595-9037-c69c656c5c38',
 'faceRectangle': {'top': 322, 'left': 223, 'width': 584, 'height': 584},
 'faceLandmarks': {'pupilLeft': {'x': 386.0, 'y': 480.7},
  'pupilRight': {'x': 641.7, 'y': 481.1},
  'noseTip': {'x': 518.0, 'y': 648.1},
  'mouthLeft': {'x': 388.9, 'y': 748.0},
  'mouthRight': {'x': 645.1, 'y': 741.8},
  'eyebrowLeftOuter': {'x': 304.3, 'y': 441.0},
  'eyebrowLeftInner': {'x': 466.8, 'y': 442.9},
  'eyeLeftOuter': {'x': 345.4, 'y': 485.3},
  'eyeLeftTop': {'x': 385.2, 'y': 464.8},
  'eyeLeftBottom': {'x': 386.5, 'y': 497.1},
  'eyeLeftInner': {'x': 424.6, 'y': 487.6},
  'eyebrowRightInner': {'x': 572.8, 'y': 448.0},
  'eyebrowRightOuter': {'x': 738.0, 'y': 445.0},
  'eyeRightInner': {'x': 603.8, 'y': 485.0},
  'eyeRightTop': {'x': 646.0, 'y': 466.2},
  'eyeRightBottom': {'x': 644.9, 'y': 496.9},
  'eyeRightOuter': {'x': 686.7, 'y': 485.3},
  'noseRootLeft': {'x': 475.1, 'y': 493.3},
  'noseRootRight': {'x': 547.5, 'y': 493.9},
  'noseLeftAlarTop': {'x': 456.7, 'y': 590.5},
  'noseRightAlarTop': {'x': 564.5, 'y': 587.1},
  'noseLeftAlarOutTip': {'x': 425.6, 'y': 643.8},
  'noseRightAlarOutTip': {'x': 595.7, 'y': 638.2},
  'upperLipTop': {'x': 524.0, 'y': 737.2},
  'upperLipBottom': {'x': 523.8, 'y': 756.8},
  'underLipTop': {'x': 522.6, 'y': 770.5},
  'underLipBottom': {'x': 525.5, 'y': 800.8}},
 'faceAttributes': {'smile': 0.999,
  'headPose': {'pitch': 0.0, 'roll': -0.4, 'yaw': 3.1},
  'gender': 'male',
  'age': 50.0,
  'facialHair': {'moustache': 0.1, 'beard': 0.1, 'sideburns': 0.1},
  'glasses': 'NoGlasses',
  'emotion': {'anger': 0.0,
   'contempt': 0.0,
   'disgust': 0.0,
   'fear': 0.0,
   'happiness': 0.999,
   'neutral': 0.001,
   'sadness': 0.0,
   'surprise': 0.0},
  'blur': {'blurLevel': 'low', 'value': 0.06},
  'exposure': {'exposureLevel': 'goodExposure', 'value': 0.71},
  'noise': {'noiseLevel': 'low', 'value': 0.09},
  'makeup': {'eyeMakeup': False, 'lipMakeup': False},
  'accessories': [],
  'occlusion': {'foreheadOccluded': False,
   'eyeOccluded': False,
   'mouthOccluded': False},
  'hair': {'bald': 0.11,
   'invisible': False,
   'hairColor': [{'color': 'brown', 'confidence': 1.0},
    {'color': 'gray', 'confidence': 0.65},
    {'color': 'blond', 'confidence': 0.36},
    {'color': 'black', 'confidence': 0.23},
    {'color': 'red', 'confidence': 0.2},
    {'color': 'other', 'confidence': 0.04}]}}}

# attribute2 = np.zeros(30000,39)
# for i,j in enumerate(labels):
#     attribute1[i][0]  = j['faceAttributes']['smile'] # 00_smile
#     attribute1[i][1]  = j['faceAttributes']['headPose']['roll'] # 01_head_Pose_Roll
#     attribute1[i][2]  = j['faceAttributes']['headPose']['yaw'] # 02_head_Pose_Yaw
#     if j['faceAttributes']['gender'] = 'male':
#         attribute1[i][3]  = 1 # 03_gender, male_female
#     else:
#         attribute1[i][3]  = 0 
#     attribute1[i][4]  = j['faceAttributes']['age'] # 04_age
#     attribute1[i][5]  = j['faceAttributes']['facialHair']['moustache'] # 05_moustache
#     attribute1[i][6]
#     attribute1[i][7]
#     attribute1[i][8]
#     attribute1[i][9]
#     attribute1[i][10]
#     attribute1[i][11]
#     attribute1[i][12]
#     attribute1[i][13]
#     attribute1[i][14]
#     attribute1[i][15]
#     attribute1[i][16]
#     attribute1[i][17]
#     attribute1[i][18]
#     attribute1[i][19]
#     attribute1[i][20]
#     attribute1[i][21]
#     attribute1[i][22]
#     attribute1[i][23]
#     attribute1[i][24]
#     attribute1[i][25]
#     attribute1[i][26]
#     attribute1[i][27]
#     attribute1[i][28]
#     attribute1[i][29]
#     attribute1[i][30]
#     attribute1[i][31]
#     attribute1[i][32]
#     attribute1[i][33]
#     attribute1[i][34]
#     attribute1[i][35]
#     attribute1[i][36]
#     attribute1[i][37]
#     attribute1[i][38]
#     attribute1[i][39]
'''

#测试潜空间包中: latent_training_data.pkl 属性的数据格式
#将其中的属性全部norm，改为 0-1的格式, 再此基础上研究:
# (1) 球状结构，属性就是一个最大特征变化的人 (a.多少个共同属性的人组成向量: 1, 10, 100, 1000? )
# (2) fc分类器去训练这个特征分类(监督，半监督，自监督)

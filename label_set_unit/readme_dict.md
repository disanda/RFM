# face label dict

## dict_v1

> note: this dict is cleaned from the file:  'latent_training_data.pkl.gz'

> there are 20,307 samples.

### clean

1. one file is the latent directions (d)

> size: (20307, 40), a sample have 40 attributes, value range [0-1]

2. one file is the corresponding latent w (face vector), size: (20307, 18, 512)
 

### origin format as:

```python
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
```

4. cleaned format dict_v1 as:

```python
# labels_dict1={ #[0-1]
#     '00_smile':
#     '01_head_Pose_Roll': 
#     '02_head_Pose_Roll_Neg': 
#     '03_head_Pose_Yaw': 
#     '04_head_Pose_Yaw_Neg':
#     '05_gender_male':0 or 1, # Male or Female
#     '06_gender_female':0 or 1, # same with above
#     '07_age':0-70,
#     '08_moustache':0.001, 
#     '09_beard':0.001,
#     '10_sideburns':0.001,
#     '11_noGlasses': 0 or 1, (1, 16396)
#     '12_readingGlasses': 0 or 1, (1, 32)
#     '13_sunglasses': 0 or 1, (1, 903)
#     '14_anger': 0-1, (<0.5,14; <0.3,32) acc=0.9993(0.5), 0.998473(0.3) #not well
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
```

## dict_v2

1. face_classifer

- tf_classifer_stylegan_torchG.py

> a attribute classifer to label faces. 

> svm classifer to (-1,1), we input softmax as range [0,1]

2. Integration

Integrate labels to a file

> integration_1.py => Integrate single label files, e.g.,  1000\*(1,40) => (1,000,40)

> integration_2.py => Integrate a small file to a large file:  100\*(1,000,40) => (10,000,40)

3. cleaned format dict_v2 as:

40 attributes

```python

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
```




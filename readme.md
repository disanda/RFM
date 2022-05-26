# A Fast Method for Real face morphing(RFM)

![RFM](./checkpoint/img/rfm.png)

> This is the official code for "Fast 2-Step Regularization on Style Optimization for Real Face Morphing"

[中文](./readme_ch.md)

## 1.Overview

The code aim to 3 contributions:

### 1.1 Label Set
    
    > we labeled large-scale latent vecotrs for 3 GANs with 40 face attributes (depend on the networks of Nvidia attibute classlifers). They are:

    > PGGAN (0-30,000), StyleGAN1: Nvdia (0-30,000)  MS (0-20,307)

    ![Stylegan1_set_clip_google_drive](https://drive.google.com/drive/folders/1Sre282bmaFDQwAOi2I0J-SdpZzuM6h83?usp=sharing)

    ![PGGAN_set_clip_google_drive](https://drive.google.com/drive/folders/1xOTXiJcoH_U6WQdwpZVwwxb5M1lLkr-i?usp=sharing)

### 2. Invert real face to style latent vector (w_y, the 1st regularization)

    > with a well trained StyleGAN encoder, refer to wy_gan_inversion.py


### 3. Find interpretable directions in style latent space (w_d, the 2nd regularization)
    >with limited labels (8,000-12,000) samples. 

Based on above, our code offered a fast way to RFM


## 2.Usage

### 2.1 Label Set

The label set at './checkpoint/label_dict/', 

Set is dict, size with (n, 40): n samples with 40 attributes


- latent vectors

[Download_'z_0_30000.pt'](https://drive.google.com/file/d/1veL5C1tbeOmXMpBj-J_oFzXRnLFCLtR6/view?usp=sharing)

> the labeled 30,000 latent vectors (from random seed id), in StyleGAN, pls use z to generate w (by M)

> you can generate z and w by youself, see './label_set_unit/generation_seed_zw.py'


- 'stylegan1_attributes_seed0_30000.pt'

> if you want to label face attributes by yourself, or other GANs.  pls refer to: './label_set_script.py'

with [Nv_face_40classifiers_tf1.14](https://drive.google.com/drive/folders/1fIDENM6TEWdIdftbEa-UkboYA-EdgU9W?usp=sharing)

- 'stylegan1_20307_attributes40_ms.pt'

> we also cleaned a Microsoft face label set 20,307 samples with 40 attributes. 

> cleaned script: './label_set_unit/label_set_ms/dict_ms_clean.py'

- the set labels z (random seed from 0 to 30,000), if StyleGAN, pls input z to make w.
>check the file: './label_set_unit/generation_seed_zw.py' to generate correspobding z and w


### 2.2 Get w_y from the 1st regularization

- pls download pre-trained model to './checkpoint'


> 3 stylegan1 models to './checkpoint/stylegan1/ffhq/'

[Google_drive_stylegan1](https://drive.google.com/drive/folders/1b87MzzOoEu8LO34AOl0AqcF6QT-sqzI9?usp=sharing)

> A encoder model to './checkpoint/stylegan1/E/'

[Google_drive_stylegan1_E](https://drive.google.com/drive/folders/1sFxht4JPC355u4UfWnK-GNdssNO0k2iM?usp=sharing)

- drag a real face (or more) to './checkpoint/real_imgs/'

> there are some faces in './checkpoint/imgs/'

> there are some w_y in './checkpoint/wy_faces'

- run the file:

> python wy_gan_inversion.py

> result will save at './result'


### 2.3 Get w_d from the 2nd regularization

> run 'wd_direction_ms.py'  if you use MS set

> run 'wd_direction_nv.py'  if you use NV set


### 2.3 RFM

> run 'rfm.py' with a learned direction

## 3. Realated work

- 3.1 GAN Encoder: https://github.com/disanda/MTV-TSA 

> This is our previous work but there some shortage. We will release a upgradedd version and a revised paper in future.

- 3.2 The other label set: https://github.com/Puzer/stylegan-encoder

> from Microsoft Classifer, labels 20,307 w with 40 attributes.

- 3.3 NV Classifier: https://github.com/NVlabs/stylegan2/blob/master/metrics/linear_separability.py

- 3.4 Baselines

a.https://github.com/genforce/interfacegan















# A Fast Method for Real face morphing(RFM)

![RFM](./checkpoint/img/rfm.png)

> This is the official code for "Fast 2-Step Regularization on Style Optimization for Real Face Morphing"

## 1.Overview

The code aim to 3 contributions:

### 1.1 Label Set
    
    > we labeled large-scale latent vecotrs for 3 GANs with 40 face attributes (depend on the networks of Nvidia attibute classlifers). They are:

    > PGGAN (0-30,000), StyleGAN1 (0-30,000)

    > if you want to label face attributes by yourself, or to other GANs.  we also release the label scipt. 

### 2. Invert real face to style latent vector (w_y, the 1st regularization)

    > with limited labels (8,000-12,000) samples. 


### 3. Find interpretable directions in style latent space (w_d, the 2nd regularization)
    >with limited labels (8,000-12,000) samples. 

- based on above, our code offered a fast way to RFM

## 2.Usage

### 2.1 Label Set

### 2.2 Get w_y from the 1st regularization

### 2.3 Get w_d from the 2nd regularization

### 2.3 RFM


## 3. Realated work

- 3.1 GAN Encoder 

> This is our previous work but there some shortage. we have upgrade it in this code. we will release a new version and revised paper about the GAN Encoder.

- 3.2 The other label set 

> from Microsoft Classifer, labels 20,307 w with 40 attributes.

> our cleaned type:

- 3.3 Baseline: InterfaceGAN, GANSpace, LatentCLR















# DCGAN-PyTorch

### Overview
This repository contains an op-for-op PyTorch reimplementation of [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](http://xxx.itp.ac.cn/pdf/1511.06434).

### Table of contents
1. [About Deep Convolutional Generative Adversarial Networks](#about-deep-convolutional-generative-adversarial-networks)
2. [Model Description](#model-description)
3. [Installation](#installation)
    * [Clone and install requirements](#clone-and-install-requirements)
    * [Download pretrained weights](#download-pretrained-weights-eg-mnist)
4. [Test](#test-eg-mnist)
5. [Train](#train)
6. [Visual](#Visual)
7. [Contributing](#contributing) 
8. [Credit](#credit)

### About Deep Convolutional Generative Adversarial Networks

If you're new to DCGAN, here's an abstract straight from the paper:

In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning. Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator. Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image representations.

### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. It receives a random noise z and generates images from this noise, which is called G(z).Discriminator is a discriminant network that discriminates whether an image is real. The input is x, x is a picture, and the output is D of x is the probability that x is a real picture, and if it's 1, it's 100% real, and if it's 0, it's not real.

### Installation

#### Clone and install requirements
```bash
$ git clone https://github.com/Lornatang/DCGAN_PyTorch.git
$ cd DCGAN_PyTorch/
$ pip3 install -r requirements.txt
```

#### Download pretrained weights (e.g MNIST)

```bash
$ cd weights/
$ bash download_weights.sh mnist
```

### Test (e.g MNIST)

Using pre training model to generate pictures.

```bash
$ python3 test.py mnist --cuda
```

### Train

```text
usage: train.py [-h] [--dataroot DATAROOT] [-j N] [--epochs N]
                [--image-size IMAGE_SIZE] [-b N] [--lr LR] [--beta1 BETA1]
                [--beta2 BETA2] [-p N] [--cuda] [--netG PATH] [--netD PATH]
                [--outf OUTF] [--manualSeed MANUALSEED] [--ngpu NGPU]
                [--multiprocessing-distributed]
                name
```

#### Example (e.g MNIST)

```bash
$ python3 train.py mnist --cuda
```

If you want to load weights that you've trained before, run the following command.

```bash
$ python3 train.py mnist --netG weights/mnist/netG_epoch_*.pth --netD weights/mnist/netD_epoch_*.pth --cuda
```

### Visual

```text
cd $REPO$/framework
sh start.sh
```

Then open the browser and type in the browser address [http://127.0.0.1:10001/](http://127.0.0.1:10001/).
Enjoy it.

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 

### Credit

#### Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

_Alec Radford, Luke Metz, Soumith Chintala_ <br>

**Abstract** <br>
In recent years, supervised learning with convolutional networks (CNNs) 
has seen huge adoption in computer vision applications. Comparatively, 
unsupervised learning with CNNs has received less attention. In this work 
we hope to help bridge the gap between the success of CNNs for supervised 
learning and unsupervised learning. We introduce a class of CNNs called deep 
convolutional generative adversarial networks (DCGANs), that have certain 
architectural constraints, and demonstrate that they are a strong candidate 
for unsupervised learning. Training on various image datasets, we show convincing 
evidence that our deep convolutional adversarial pair learns a hierarchy of 
representations from object parts to scenes in both the generator and discriminator. 
Additionally, we use the learned features for novel tasks - demonstrating their 
applicability as general image representations.

[[Paper]](https://arxiv.org/abs/1511.06434)) [[Authors' Implementation]](https://github.com/Newmu/dcgan_code)

```
@inproceedings{ICLR 2016,
  title={Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks},
  author={Alec Radford, Luke Metz, Soumith Chintala},
  booktitle={Under review as a conference paper at ICLR 2016},
  year={2016}
}
```
# DCGAN-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation
of [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](http://xxx.itp.ac.cn/pdf/1511.06434)
.

### Table of contents

1. [About Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](#about-unsupervised-representation-learning-with-deep-convolutional-generative-adversarial-networks)
2. [Model Description](#model-description)
3. [Installation](#installation)
    * [Clone and install requirements](#clone-and-install-requirements)
    * [Download pretrained weights](#download-pretrained-weights-eg-cifar10)
    * [Download cartoon faces](#download-cartoon-faces)
4. [Test](#test)
5. [Train](#train-eg-cifar10)
6. [Contributing](#contributing)
7. [Credit](#credit)

### About Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

If you're new to DCGAN, here's an abstract straight from the paper:

In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision
applications. Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help
bridge the gap between the success of CNNs for supervised learning and unsupervised learning. We introduce a class of
CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain architectural constraints,
and demonstrate that they are a strong candidate for unsupervised learning. Training on various image datasets, we show
convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts
to scenes in both the generator and discriminator. Additionally, we use the learned features for novel tasks -
demonstrating their applicability as general image representations.

### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. It receives
a random noise z and generates images from this noise, which is called G(z).Discriminator is a discriminant network that
discriminates whether an image is real. The input is x, x is a picture, and the output is D of x is the probability that
x is a real picture, and if it's 1, it's 100% real, and if it's 0, it's not real.

### Installation

#### Clone and install requirements

```shell
$ git clone https://github.com/Lornatang/DCGAN-PyTorch.git
$ cd DCGAN-PyTorch/
$ pip3 install -r requirements.txt
```

#### Download pretrained weights (e.g. CIFAR10)

```shell
$ cd weights/
$ python3 download_weights.py
```

#### Download cartoon faces

[baiduclouddisk](https://pan.baidu.com/s/1nawrN1Kiw3Z2Jk1NgJqZTQ)  access: `68rn`

### Test

Using pre training model to generate pictures.

```text
usage: test.py [-h] [-a ARCH] [-n NUM_IMAGES] [--outf PATH] [--device DEVICE]

Research and application of GAN based super resolution technology for
pathological microscopic images.

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: _gan | cifar10 (default: cifar10).
  -n NUM_IMAGES, --num-images NUM_IMAGES
                        How many samples are generated at one time. (default:
                        64).
  --outf PATH           The location of the image in the evaluation process.
                        (default: ``test``).
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default:
                        ``cpu``).

# Example (e.g. CIFAR10)
$ python3 test.py -a cifar10
```

<span align="center"><img src="assets/cifar10.gif" alt="">
</span>

### Train (e.g. CIFAR10)

```text
usage: train.py [-h] --dataset DATASET [--dataroot DATAROOT] [-j N]
                [--manualSeed MANUALSEED] [--device DEVICE] [-p N] [-a ARCH]
                [--model-path PATH] [--pretrained] [--netD PATH] [--netG PATH]
                [--start-epoch N] [--iters N] [-b N] [--image-size IMAGE_SIZE]
                [--channels CHANNELS] [--lr LR]

Research and application of GAN based super resolution technology for
pathological microscopic images.

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     cifar10 | cartoon.
  --dataroot DATAROOT   Path to dataset. (default: ``data``).
  -j N, --workers N     Number of data loading workers. (default:4)
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:1111)
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default: ````).
  -p N, --save-freq N   Save frequency. (default: 50).
  -a ARCH, --arch ARCH  model architecture: cifar10 | cartoon (default: mnist).
  --model-path PATH     Path to latest checkpoint for model. (default: ````).
  --pretrained          Use pre-trained model.
  --netD PATH           Path to latest discriminator checkpoint. (default:
                        ````).
  --netG PATH           Path to latest generator checkpoint. (default: ````).
  --start-epoch N       manual epoch number (useful on restarts)
  --iters N             The number of iterations is needed in the training of
                        PSNR model. (default: 2e4)
  -b N, --batch-size N  mini-batch size (default: 64), this is the total batch
                        size of all GPUs on the current node when using Data
                        Parallel or Distributed Data Parallel.
  --image-size IMAGE_SIZE
                        The height / width of the input image to network.
                        (default: 64).
  --lr LR               Learning rate. (default:2e-4)

# Example (e.g. CIFAR10)
$ python3 train.py -a cifar10 --dataset cifar10 --image-size 64 --pretrained
```

If you want to load weights that you've trained before, run the following command.

```bash
$ python3 train.py -a cifar10 \
                   --dataset cifar10 \
                   --image-size 64 \
                   --start-epoch 18 \
                   --netG weights/netG_epoch_18.pth \
                   --netD weights/netD_epoch_18.pth
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

_Alec Radford, Luke Metz, Soumith Chintala_ <br>

**Abstract** <br>
In recent years, supervised learning with convolutional networks (CNNs)
has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less
attention. In this work we hope to help bridge the gap between the success of CNNs for supervised learning and
unsupervised learning. We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs),
that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning.
Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a
hierarchy of representations from object parts to scenes in both the generator and discriminator. Additionally, we use
the learned features for novel tasks - demonstrating their applicability as general image representations.

[[Paper]](https://arxiv.org/abs/1511.06434)) [[Authors' Implementation]](https://github.com/Newmu/dcgan_code)

```
@inproceedings{ICLR 2016,
  title={Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks},
  author={Alec Radford, Luke Metz, Soumith Chintala},
  booktitle={Under review as a conference paper at ICLR 2016},
  year={2016}
}
```
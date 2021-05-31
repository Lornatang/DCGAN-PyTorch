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
    * [Download pretrained weights](#download-pretrained-weights-eg-imagenet)
    * [Download cartoon faces](#download-cartoon-faces)
4. [Test](#test)
    * [Torch Hub call](#torch-hub-call)
    * [Base call](#base-call)
5. [Train](#train-eg-imagenet)
6. [Contributing](#contributing)
7. [Credit](#credit)

### About Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

If you're new to DCGAN, here's an abstract straight from the paper:

In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. Comparatively,
unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between the success of CNNs for supervised
learning and unsupervised learning. We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain
architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning. Training on various image datasets, we show
convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to scenes in both the
generator and discriminator. Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image
representations.

### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. It receives a random noise z and generates
images from this noise, which is called G(z).Discriminator is a discriminant network that discriminates whether an image is real. The input is x, x is
a picture, and the output is D of x is the probability that x is a real picture, and if it's 1, it's 100% real, and if it's 0, it's not real.

### Installation

#### Clone and install requirements

```shell
$ git clone https://github.com/Lornatang/DCGAN-PyTorch.git
$ cd DCGAN-PyTorch/
$ pip3 install -r requirements.txt
```

#### Download pretrained weights (e.g. ImageNet)

```shell
$ cd weights/
$ python3 download_weights.py
```

#### Download cartoon faces

[baidu cloud disk](https://pan.baidu.com/s/1nawrN1Kiw3Z2Jk1NgJqZTQ)  access: `68rn`

### Test

#### Torch hub call

```python
# Using Torch Hub library.
import torch
import torchvision.utils as vutils

# Choose to use the device.
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the model into the specified device.
model = torch.hub.load("Lornatang/DCGAN-PyTorch", "dcgan", pretrained=True, progress=True, verbose=False)
model.eval()
model = model.to(device)

# Create random noise image.
num_images = 64
noise = torch.randn([num_images, 100, 1, 1], device=device)

# The noise is input into the generator model to generate the image.
with torch.no_grad():
    generated_images = model(noise)

# Save generate image.
vutils.save_image(generated_images, "image.png", normalize=True)
```

#### Base call

```text
usage: test.py [-h] [-a ARCH] [--num-images NUM_IMAGES] [--model-path PATH] [--pretrained] [--seed SEED] [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: dcgan. (Default: `dcgan`)
  --num-images NUM_IMAGES
                        How many samples are generated at one time. (Default: 64)
  --model-path PATH     Path to latest checkpoint for model.
  --pretrained          Use pre-trained model.
  --seed SEED           Seed for initializing training.
  --gpu GPU             GPU id to use.

# Example (e.g. ImageNet)
$ python3 test.py -a dcgan --pretrained --gpu 0 
```

<span align="center"><img src="assets/mnist.gif" alt="">
</span>

### Train (e.g. ImageNet)

```text
usage: train.py [-h] [-a ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N] [--lr LR] [--image-size IMAGE_SIZE] [--channels CHANNELS] [--netD PATH] [--netG PATH] [--pretrained] [--world-size WORLD_SIZE] [--rank RANK] [--dist-url DIST_URL]
                [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU] [--multiprocessing-distributed]
                DIR

positional arguments:
  DIR                   Path to dataset.

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  Model architecture: gan. (Default: gan)
  -j N, --workers N     Number of data loading workers. (Default: 4)
  --epochs N            Number of total epochs to run. (Default: 128)
  --start-epoch N       Manual epoch number (useful on restarts). (Default: 0)
  -b N, --batch-size N  Mini-batch size (default: 64), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel.
  --lr LR               Learning rate. (Default: 0.0002)
  --image-size IMAGE_SIZE
                        Image size of high resolution image. (Default: 28)
  --channels CHANNELS   The number of channels of the image. (Default: 1)
  --netD PATH           Path to Discriminator checkpoint.
  --netG PATH           Path to Generator checkpoint.
  --pretrained          Use pre-trained model.
  --world-size WORLD_SIZE
                        Number of nodes for distributed training.
  --rank RANK           Node rank for distributed training. (Default: -1)
  --dist-url DIST_URL   url used to set up distributed training. (Default: `tcp://59.110.31.55:12345`)
  --dist-backend DIST_BACKEND
                        Distributed backend. (Default: `nccl`)
  --seed SEED           Seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training.

# Example (e.g. ImageNet)
$ python3 train.py -a dcgan --gpu 0 data
```

If you want to load weights that you've trained before, run the following command.

```bash
$ python3 train.py -a dcgan --netD weights/Discriminator_epoch8.pth --netG weights/Generator_epoch8.pth --start-epoch 8 --gpu 0 data
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.
I look forward to seeing what the community does with these models!

### Credit

#### Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

_Alec Radford, Luke Metz, Soumith Chintala_ <br>

**Abstract** <br>
In recent years, supervised learning with convolutional networks (CNNs)
has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less attention. In this work we
hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. We introduce a class of CNNs called deep
convolutional generative adversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate
for unsupervised learning. Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a
hierarchy of representations from object parts to scenes in both the generator and discriminator. Additionally, we use the learned features for novel
tasks - demonstrating their applicability as general image representations.

[[Paper]](https://arxiv.org/abs/1511.06434)) [[Authors' Implementation]](https://github.com/Newmu/dcgan_code)

```
@inproceedings{ICLR 2016,
  title={Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks},
  author={Alec Radford, Luke Metz, Soumith Chintala},
  booktitle={Under review as a conference paper at ICLR 2016},
  year={2016}
}
```

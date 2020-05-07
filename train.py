# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""In recent years, supervised learning with convolutional networks (CNNs) has
seen huge adoption in computer vision applications. Comparatively, unsupervised
learning with CNNs has received less attention. In this work we hope to
help bridge the gap between the success of CNNs for supervised learning and
unsupervised learning. We introduce a class of CNNs called deep convolutional
generative adversarial networks (DCGANs), that have certain architectural
constraints, and demonstrate that they are a strong candidate for unsupervised
learning. Training on various image datasets, we show convincing evidence that
our deep convolutional adversarial pair learns a hierarchy of representations
from object parts to scenes in both the generator and discriminator.
Additionally, we use the learned features for novel tasks - demonstrating their
applicability as general image representations.
"""
import argparse
import os
import random
import warnings

import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

from dcgan_pytorch import Discriminator
from dcgan_pytorch import Generator
from dcgan_pytorch import weights_init

parser = argparse.ArgumentParser(description="PyTorch simple implementation Deep Convolutional GANs")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="path to datasets. (default:`./data`)")
parser.add_argument("name", type=str,
                    help="dataset name. Option: [mnist, cifar, imagenet]")
parser.add_argument("-j", "--workers", default=8, type=int, metavar="N",
                    help="Number of data loading workers. (default:8)")
parser.add_argument("--epochs", default=25, type=int, metavar="N",
                    help="Number of total epochs to run. (default:25)")
parser.add_argument("--image-size", type=int, default=64,
                    help="Size of the data crop (squared assumed). (default:64)")
parser.add_argument("-b", "--batch-size", default=64, type=int,
                    metavar="N",
                    help="mini-batch size (default: 64), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="learning rate. (default=0.0002)")
parser.add_argument("--beta1", type=float, default=0.5,
                    help="beta1 for adam. (default:0.5)")
parser.add_argument("--beta2", type=float, default=0.999,
                    help="beta2 for adam. (default:0.999)")
parser.add_argument("-p", "--print-freq", default=100, type=int,
                    metavar="N", help="print frequency (default:100)")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--netG", default="", type=str, metavar="PATH",
                    help="path to latest generator checkpoint (default: none)")
parser.add_argument("--netD", default="", type=str, metavar="PATH",
                    help="path to latest discriminator checkpoint (default: none)")
parser.add_argument("--outf", default="./outputs",
                    help="folder to output images. (default:`./outputs`).")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")
parser.add_argument("--ngpu", default=1, type=int,
                    help="GPU id to use. (default:None)")
parser.add_argument("--multiprocessing-distributed", action="store_true",
                    help="Use multi-processing distributed training to launch "
                         "N processes per node, which has N GPUs. This is the "
                         "fastest way to use PyTorch for either single node or "
                         "multi node data parallel training")

fixed_noise = torch.randn(64, 100, 1, 1)

valid_dataset_name = ["mnist", "cifar", "imagenet"]

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs("weights")
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Dataset
if args.name == "imagenet":
    dataset = datasets.ImageFolder(root=args.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(64),
                                       transforms.CenterCrop(64),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    nc = 3
elif args.name == "cifar":
    dataset = datasets.CIFAR10(root=args.dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(64),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nc = 3
elif args.name == "mnist":
    dataset = datasets.MNIST(root=args.dataroot, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(64),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,)),
                             ]))
    nc = 1
else:
    warnings.warn("You have chosen a specific dataset. This will default use MNIST dataset!")
    dataset = datasets.MNIST(root=args.dataroot, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(64),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,)),
                             ]))
    nc = 1
    try:
        os.makedirs(os.path.join(args.outf, "mnist"))
    except OSError:
        pass

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, pin_memory=True, num_workers=int(args.workers))

assert len(dataloader) > 0, f"WARNING: Please check that your dataset name. Option: {valid_dataset_name}"

try:
    os.makedirs(os.path.join(args.outf, str(args.name)))
except OSError:
    pass

try:
    os.makedirs(os.path.join("weights", str(args.name)))
except OSError:
    pass

device = torch.device("cuda:0" if args.cuda else "cpu")
ngpu = int(args.ngpu)

# create model
netG = Generator(nc).to(device)
netD = Discriminator(nc).to(device)

if args.cuda and ngpu > 1 and args.batch_size > 1:
    netG = torch.nn.DataParallel(netG).to(device)
    netD = torch.nn.DataParallel(netD).to(device)

netG.apply(weights_init)
netD.apply(weights_init)

if args.netG != "":
    netG.load_state_dict(torch.load(args.netG))
if args.netD != "":
    netD.load_state_dict(torch.load(args.netD))

# define loss function (adversarial_loss) and optimizer
adversarial_loss = torch.nn.BCELoss().to(device)
optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

g_losses = []
d_losses = []

for epoch in range(0, args.epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, data in progress_bar:
        # get batch size data
        real_images = data[0].to(device)
        batch_size = real_images.size(0)

        # real data label is 1, fake data label is 0.
        real_label = torch.full((batch_size,), 1, device=device, dtype=torch.float32)
        fake_label = torch.full((batch_size,), 0, device=device, dtype=torch.float32)

        # Sample noise as generator input
        noise = torch.randn(batch_size, 100, 1, 1, device=device)

        ##############################################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ##############################################
        netD.zero_grad()

        # Train with real
        real_output = netD(real_images)
        errD_real = adversarial_loss(real_output, real_label)
        D_x = real_output.mean().item()

        # Generate fake image batch with G
        fake = netG(noise)

        # Train with fake
        fake_output = netD(fake.detach())
        errD_fake = adversarial_loss(fake_output, fake_label)
        D_G_z1 = fake_output.mean().item()

        # Add the gradients from the all-real and all-fake batches
        errD = (errD_real + errD_fake) / 2
        errD.backward()
        # Update D
        optimizer_D.step()

        ##############################################
        # (2) Update G network: maximize log(D(G(z)))
        ##############################################
        netG.zero_grad()

        fake_output = netD(fake)
        errG = adversarial_loss(fake_output, real_label)
        errG.backward()
        D_G_z2 = fake_output.mean().item()
        # Update G
        optimizer_G.step()

        progress_bar.set_description(
            f"[{epoch}/{args.epochs}][{i}/{len(dataloader)}] "
            f"Loss_D: {errD.item():.4f} "
            f"Loss_G: {errG.item():.4f} "
            f"D_x: {D_x:.4f} "
            f"D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}")

        if i % args.print_freq == 0:
            vutils.save_image(real_images, f"{args.outf}/{args.name}/real_samples.png", normalize=True)

            # Save Losses for plotting later
            g_losses.append(errG.item())
            d_losses.append(errD.item())

            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(), f"{args.outf}/{args.name}/fake_samples_epoch_{epoch}.png", normalize=True)

        # do check pointing
        torch.save(netG.state_dict(), f"weights/{args.name}/netG_epoch_{epoch}.pth")
        torch.save(netD.state_dict(), f"weights/{args.name}/netD_epoch_{epoch}.pth")

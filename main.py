# Copyright 2019 Lorna Authors. All Rights Reserved.
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

"""Generative Adversarial Networks (GANs) are one of the most interesting ideas
in computer science today. Two models are trained simultaneously by
an adversarial process. A generator ("the artist") learns to create images
that look real, while a discriminator ("the art critic") learns
to tell real images apart from fakes.
"""

import argparse
import os
import random

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from model import Discriminator
from model import Generator
from model import weights_init

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='./datasets', help='path to datasets')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='inputs batch size')
parser.add_argument('--img_size', type=int, default=64, help='the height / width of the inputs image to network')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--epochs', type=int, default=200, help="Train loop")
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./imgs', help='folder to output images')
parser.add_argument('--checkpoint_dir', default='./checkpoints', help='folder to output checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--phase', type=str, default='train', help='model mode. default=`train`, option=`generate`')

opt = parser.parse_args()

try:
  os.makedirs(opt.outf)
except OSError:
  pass

if opt.manualSeed is None:
  opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ngpu = int(opt.ngpu)

fixed_noise = torch.randn(opt.batch_size, 100, 1, 1, device=device)


def train():
  """ train model
  """
  try:
    os.makedirs(f"{opt.checkpoint_dir}")
  except OSError:
    pass
  ################################################
  #               load train dataset
  ################################################
  dataset = dset.ImageFolder(root=opt.dataroot,
                             transform=transforms.Compose([
                               transforms.Resize(opt.img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ]))

  assert dataset
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                           shuffle=True, num_workers=int(opt.workers))

  ################################################
  #               load model
  ################################################
  netG = Generator(ngpu).to(device)
  netG.apply(weights_init)
  if opt.netG != "":
    netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, loc: storage))
  print(netG)

  netD = Discriminator(ngpu).to(device)
  netD.apply(weights_init)
  if opt.netD != "":
    netD.load_state_dict(torch.load(opt.netD, map_location=lambda storage, loc: storage))
  print(netD)

  ################################################
  #           Binary Cross Entropy
  ################################################
  criterion = nn.BCELoss().to(device)

  ################################################
  #            Use Adam optimizer
  ################################################
  optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
  optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

  ################################################
  #               print args
  ################################################
  print("########################################")
  print(f"train datasets path: {opt.dataroot}")
  print(f"work thread: {opt.workers}")
  print(f"batch size:{opt.batch_size}")
  print(f"Epochs: {opt.epochs}")
  print("########################################")
  print("Starting trainning!")
  for epoch in range(opt.epochs):
    for i, data in enumerate(dataloader):
      ##############################################
      # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      ##############################################
      # train with real
      netD.zero_grad()
      real_data = data[0].to(device)
      batch_size = real_data.size(0)

      # real data label is 1, fake data label is 0.
      real_label = torch.full((batch_size,), 1, device=device)
      fake_label = torch.full((batch_size,), 0, device=device)

      output = netD(real_data)
      errD_real = criterion(output, real_label)
      errD_real.backward()

      # train with fake
      noise = torch.randn(batch_size, 100, 1, 1, device=device)
      fake = netG(noise)
      output = netD(fake.detach())
      errD_fake = criterion(output, fake_label)
      errD_fake.backward()
      errD = errD_real + errD_fake
      optimizerD.step()

      ##############################################
      # (2) Update G network: maximize log(D(G(z)))
      ##############################################
      netG.zero_grad()
      output = netD(fake)
      errG = criterion(output, real_label)
      errG.backward()
      optimizerG.step()
      if i % 20 == 0:
        print(f"Epoch->[{epoch + 1:3d}/{opt.epochs}] "
              f"Progress->{i / len(dataloader) * 100:4.2f}% "
              f"Loss_D: {errD.item():.4f} "
              f"Loss_G: {errG.item():.4f} ", end="\r")

      if i % 100 == 0:
        vutils.save_image(real_data, f"{opt.outf}/real_samples.png", normalize=True)
        fake = netG(fixed_noise)
        vutils.save_image(fake.detach(), f"{opt.outf}/fake_samples_epoch_{epoch + 1}.png", normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), f"{opt.checkpoint_dir}/G.pth")
    torch.save(netD.state_dict(), f"{opt.checkpoint_dir}/D.pth")


def generate():
  """ random generate fake image.
  """
  ################################################
  #               load model
  ################################################
  print(f"Load model...\n")
  netG = Generator(ngpu).to(device)
  if opt.netG != "":
    netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, loc: storage))
  print(f"Load model successful!")
  with torch.no_grad():
    for i in range(64):
      z = torch.randn(1, opt.nz, 1, 1, device=device)
      fake = netG(z)
      vutils.save_image(fake.detach(), f"unknown/fake_{i + 1:04d}.png", normalize=True)
  print("Images have been generated!")


if __name__ == '__main__':
  if opt.phase == 'train':
    train()
  elif opt.phase == 'generate':
    generate()
  else:
    print(opt)

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
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from model import Discriminator
from model import Generator

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='./datasets', help='path to datasets')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--out_images', default='./imgs', help='folder to output images')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--phase', type=str, default='train', help='model mode. default=`train`, option=`generate`')

opt = parser.parse_args()

try:
  os.makedirs(opt.out_images)
except OSError:
  pass

if opt.manualSeed is None:
  opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fixed_noise = torch.randn(64, 100, 1, 1, device=device)


def train():
  """ train model
  """
  try:
    os.makedirs("./checkpoints")
  except OSError:
    pass
  ################################################
  #               load train dataset
  ################################################
  dataset = dset.ImageFolder(root=opt.dataroot,
                             transform=transforms.Compose([
                               transforms.Resize(96),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ]))

  assert dataset
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                           shuffle=True, num_workers=int(opt.workers))

  ################################################
  #               load model
  ################################################
  if torch.cuda.device_count() > 1:
    netG = torch.nn.DataParallel(Generator())
  else:
    netG = Generator()
  if os.path.exists("./checkpoints/G.pth"):
    netG.load_state_dict(torch.load("./checkpoints/G.pth", map_location=lambda storage, loc: storage))

  if torch.cuda.device_count() > 1:
    netD = torch.nn.DataParallel(Discriminator())
  else:
    netD = Discriminator()
  if os.path.exists("./checkpoints/D.pth"):
    netD.load_state_dict(torch.load("./checkpoints/D.pth", map_location=lambda storage, loc: storage))

  netG.train()
  netG.to(device)
  netD.train()
  netD.to(device)
  print(netG)
  print(netD)

  ################################################
  #           Binary Cross Entropy
  ################################################
  criterion = nn.BCEWithLogitsLoss()

  ################################################
  #            Use Adam optimizer
  ################################################
  optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
  optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

  ################################################
  #               print args
  ################################################
  print("########################################")
  print(f"train dataset path: {opt.dataroot}")
  print(f"work thread: {opt.workers}")
  print(f"batch size: 64")
  print(f"image size: 96")
  print(f"Epochs: 200")
  print(f"Noise size: 100")
  print("########################################")
  print(f"loading pretrain model successful!\n")
  print("Starting trainning!")
  for epoch in range(200):
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

      output = netD(real_data).view(-1)
      errD_real = criterion(output, real_label)
      errD_real.backward()
      D_x = output.mean().item()

      # train with fake
      noise = torch.randn(batch_size, 100, 1, 1, device=device)
      fake = netG(noise)
      output = netD(fake.detach()).view(-1)
      errD_fake = criterion(output, fake_label)
      errD_fake.backward()
      D_G_z1 = output.mean().item()
      errD = errD_real + errD_fake
      optimizerD.step()

      ##############################################
      # (2) Update G network: maximize log(D(G(z)))
      ##############################################
      netG.zero_grad()
      output = netD(fake).view(-1)
      errG = criterion(output, real_label)
      errG.backward()
      D_G_z2 = output.mean().item()
      optimizerG.step()
      if i % 400 == 0:
        print(f"Epoch->[{epoch + 1:3d}/50] "
              f"Progress->{i / len(dataloader) * 100:4.2f}% "
              f"Loss_D: {errD.item():.4f} "
              f"Loss_G: {errG.item():.4f} "
              f"D(x): {D_x:.4f} "
              f"D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}")

      if i % 400 == 0:
        vutils.save_image(real_data, f"{opt.out_images}/real_samples.png", normalize=True)
        with torch.no_grad():
          fake = netG(fixed_noise).detach().cpu()
        vutils.save_image(fake, f"{opt.out_images}/fake_samples_epoch_{epoch + 1}.png", normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), f"{opt.out_folder}/G.pth")
    torch.save(netD.state_dict(), f"{opt.out_folder}/D.pth")


if __name__ == '__main__':
  if opt.phase == 'train':
    train()
  elif opt.phase == 'generate':
    pass
  else:
    print(opt)

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
import torch
import torch.nn as nn


# Generative Adversarial Networks model architecture from
# the One weird trick...
# <https://arxiv.org/abs/1511.06434>`_ paper.
class Generator(nn.Module):
    """

    Examples:
      >>> import torch
      >>> from dcgan_pytorch import Generator
      >>> from dcgan_pytorch import Discriminator
      >>> generator = Generator()
      >>> discriminator = Discriminator()
      >>> generator.eval()
      >>> noise = torch.randn(1, 100, 1, 1)
      >>> discriminator(generator(noise)).item()
    """

    def __init__(self, channels=3):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # inputs is Z, going into a convolution
            nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, inputs):
        outputs = self.main(inputs)
        return outputs


class Discriminator(nn.Module):
    """

    Examples:
      >>> import torch
      >>> from dcgan_pytorch import Discriminator
      >>> discriminator = Discriminator()
      >>> discriminator.eval()
      >>> noise = torch.randn(1, 784, 1, 1)
      >>> discriminator(noise).item()
    """

    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # inputs is (nc) x 64 x 64
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        outputs = self.main(inputs)
        outputs = torch.flatten(outputs)
        return outputs

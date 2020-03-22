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

from .utils import get_model_params
from .utils import load_pretrained_weights
from .utils import model_params


# Generative Adversarial Networks model architecture from
# the One weird trick...
# <https://arxiv.org/abs/1511.06434>`_ paper.
class Generator(nn.Module):
    r""" An Generator model. Most easily loaded with the .from_name
    or .from_pretrained methods

    Args:
      global_params (namedtuple): A set of GlobalParams shared between blocks

    Examples:
      >>> import torch
      >>> from dcgan_pytorch import Generator
      >>> from dcgan_pytorch import Discriminator
      >>> generator = Generator.from_pretrained("g-mnist")
      >>> discriminator = Discriminator.from_pretrained("g-mnist")
      >>> generator.eval()
      >>> noise = torch.randn(1, 100, 1, 1)
      >>> discriminator(generator(noise)).item()
    """

    def __init__(self, global_params=None):
        super(Generator, self).__init__()
        self.noise = global_params.noise
        self.channels = global_params.channels

        self.main = nn.Sequential(
            # inputs is Z, going into a convolution
            nn.ConvTranspose2d(self.noise, 64 * 8, 4, 1, 0, bias=False),
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
            nn.ConvTranspose2d(64, self.channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, inputs):
        outputs = self.main(inputs)
        return outputs

    @classmethod
    def from_name(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        global_params = get_model_params(model_name)
        return cls(global_params)

    @classmethod
    def from_pretrained(cls, model_name):
        model = cls.from_name(model_name, )
        load_pretrained_weights(model, model_name)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, res = model_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_list = ["mnist", "fmnist", "cifar", "imagenet"]
        valid_models = ["g-" + str(i) for i in valid_list]
        if model_name not in valid_models:
            raise ValueError(
                "model_name should be one of: " + ", ".join(valid_models))


class Discriminator(nn.Module):
    r""" An Discriminator model. Most easily loaded with the .from_name or
    .from_pretrained methods

    Args:
      global_params (namedtuple): A set of GlobalParams shared between blocks

    Examples:
      >>> import torch
      >>> from dcgan_pytorch import Discriminator
      >>> discriminator = Discriminator.from_pretrained("d-mnist")
      >>> discriminator.eval()
      >>> noise = torch.randn(1, 784, 1, 1)
      >>> discriminator(noise).item()
    """

    def __init__(self, global_params=None):
        super(Discriminator, self).__init__()
        self.channels = global_params.channels
        self.image_size = global_params.image_size
        self.negative_slope = global_params.negative_slope

        self.main = nn.Sequential(
            # inputs is (nc) x 64 x 64
            nn.Conv2d(self.channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(self.negative_slope, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(self.negative_slope, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(self.negative_slope, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(self.negative_slope, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        outputs = self.main(inputs)
        outputs = torch.flatten(outputs)
        return outputs

    @classmethod
    def from_name(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        global_params = get_model_params(model_name)
        return cls(global_params)

    @classmethod
    def from_pretrained(cls, model_name):
        model = cls.from_name(model_name, )
        load_pretrained_weights(model, model_name)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, res = model_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_list = ["mnist", "fmnist", "cifar", "imagenet"]
        valid_models = ["d-" + str(i) for i in valid_list]
        if model_name not in valid_models:
            raise ValueError(
                "model_name should be one of: " + ", ".join(valid_models))

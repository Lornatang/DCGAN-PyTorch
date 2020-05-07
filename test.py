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
import argparse
import random

import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.utils as vutils

from dcgan_pytorch import Generator

parser = argparse.ArgumentParser(description="PyTorch simple implementation Deep Convolutional GANs")
parser.add_argument("--nc", default=3, type=int, help="Image channels. (default:3)")
parser.add_argument("--ngpu", default=1, type=int,
                    help="GPU id to use. (default:None)")
parser.add_argument("--netG", default="", type=str, metavar="PATH",
                    help="path to latest generator checkpoint (default: none)")
parser.add_argument("--outf", default="./outputs",
                    help="folder to output images. (default:`./outputs`).")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")

args = parser.parse_args()
print(args)

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

noise = torch.randn(64, 100, 1, 1, device=device)

model = Generator(args.nc).to(device)

if int(args.ngpu) > 1:
    model = torch.nn.DataParallel(model).to(device)

if args.netG != "":
    model.load_state_dict(torch.load(args.netG))

with torch.no_grad():
    fake = model(noise)
    vutils.save_image(fake.detach(), f"{args.outf}/fake.png", normalize=True)
print("The fake image has been generated!")

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


import torch
import torchvision.utils as vutils

from model import Generator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate():
  """ random generate fake image.
  """
  ################################################
  #               load model
  ################################################
  print(f"Load model...\n")
  if torch.cuda.device_count() > 1:
    netG = torch.nn.DataParallel(Generator())
  else:
    netG = Generator()
  netG.to(device)
  netG.load_state_dict(torch.load("./checkpoints/G.pth", map_location=lambda storage, loc: storage))
  netG.eval()
  print(f"Load model successful!")
  with torch.no_grad():
    z = torch.randn(1, 100, 1, 1, device=device)
    fake = netG(z).detach().cpu()
    vutils.save_image(fake, f"./static/cartoon_sister.png", normalize=True)
  print("Images have been generated!")

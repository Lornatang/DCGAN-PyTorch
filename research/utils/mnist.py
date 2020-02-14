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

"""This is mainly used to generate new training sets."""

import codecs
import os
import warnings

import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image

from main import Generator
from utils.visual import VisionDataset
from .utils import download_and_extract_archive
from .utils import makedir_exist_ok


class MNIST(VisionDataset):
  """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

  Args:
      root (string): Root directory of dataset where ``MNIST/processed/training.pt``
          and  ``MNIST/processed/test.pt`` exist.
      train (bool, optional): If True, creates dataset from ``training.pt``,
          otherwise from ``test.pt``.
      download (bool, optional): If true, downloads the dataset from the internet and
          puts it in root directory. If dataset is already downloaded, it is not
          downloaded again.
      transform (callable, optional): A function/transform that  takes in an PIL image
          and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the
          target and transforms it.
  """
  urls = [
    'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
  ]
  training_file = 'training.pt'
  test_file = 'test.pt'
  classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
             '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

  @property
  def train_labels(self):
    warnings.warn("train_labels has been renamed targets")
    return self.targets

  @property
  def test_labels(self):
    warnings.warn("test_labels has been renamed targets")
    return self.targets

  @property
  def train_data(self):
    warnings.warn("train_data has been renamed data")
    return self.data

  @property
  def test_data(self):
    warnings.warn("test_data has been renamed data")
    return self.data

  def __init__(self, root, train=True, transform=None, target_transform=None,
               download=False):
    super(MNIST, self).__init__(root, transform=transform,
                                target_transform=target_transform)
    self.train = train  # training set or test set

    if download:
      self.download()

    if not self._check_exists():
      raise RuntimeError('Dataset not found.' +
                         ' You can use download=True to download it')

    if self.train:
      data_file = self.training_file
    else:
      data_file = self.test_file
    self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data[index], int(self.targets[index])

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(img.numpy(), mode='L')

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.data)

  @property
  def raw_folder(self):
    return os.path.join(self.root, self.__class__.__name__, 'raw')

  @property
  def processed_folder(self):
    return os.path.join(self.root, self.__class__.__name__, 'processed')

  @property
  def class_to_idx(self):
    return {_class: i for i, _class in enumerate(self.classes)}

  def _check_exists(self):
    return (os.path.exists(os.path.join(self.processed_folder,
                                        self.training_file)) and
            os.path.exists(os.path.join(self.processed_folder,
                                        self.test_file)))

  def download(self):
    """Download the MNIST data if it doesn't exist in processed_folder already."""

    if self._check_exists():
      return

    makedir_exist_ok(self.raw_folder)
    makedir_exist_ok(self.processed_folder)

    # download files
    for url in self.urls:
      filename = url.rpartition('/')[2]
      download_and_extract_archive(url, download_root=self.raw_folder, filename=filename)

    # process and save as torch files
    print('Processing...')

    training_set = (
      read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
      read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
    )
    test_set = (
      read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
      read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
    )
    with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
      torch.save(training_set, f)
    with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
      torch.save(test_set, f)

    print('Done!')

  def extra_repr(self):
    return "Split: {}".format("Train" if self.train is True else "Test")


def read_sn3_pascalvincent_tensor(path, strict=True):
  """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
     Argument may be a filename, compressed filename, or file object.
  """
  # typemap
  if not hasattr(read_sn3_pascalvincent_tensor, 'typemap'):
    read_sn3_pascalvincent_tensor.typemap = {
      8: (torch.uint8, np.uint8, np.uint8),
      9: (torch.int8, np.int8, np.int8),
      11: (torch.int16, np.dtype('>i2'), 'i2'),
      12: (torch.int32, np.dtype('>i4'), 'i4'),
      13: (torch.float32, np.dtype('>f4'), 'f4'),
      14: (torch.float64, np.dtype('>f8'), 'f8')}
  # read
  with open_maybe_compressed_file(path) as f:
    data = f.read()
  # parse
  magic = get_int(data[0:4])
  nd = magic % 256
  ty = magic // 256
  assert 1 <= nd <= 3
  assert 8 <= ty <= 14
  m = read_sn3_pascalvincent_tensor.typemap[ty]
  s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
  parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
  assert parsed.shape[0] == np.prod(s) or not strict
  return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


def get_int(b):
  return int(codecs.encode(b, 'hex'), 16)


def open_maybe_compressed_file(path):
  """Return a file object that possibly decompresses 'path' on the fly.
     Decompression occurs when argument `path` is a string and ends with '.gz' or '.xz'.
  """
  if not isinstance(path, torch._six.string_classes):
    return path
  if path.endswith('.gz'):
    import gzip
    return gzip.open(path, 'rb')
  if path.endswith('.xz'):
    import lzma
    return lzma.open(path, 'rb')
  return open(path, 'rb')


def read_label_file(path):
  with open(path, 'rb') as f:
    x = read_sn3_pascalvincent_tensor(f, strict=False)
  assert (x.dtype == torch.uint8)
  assert (x.ndimension() == 1)
  return x.long()


def read_image_file(path):
  with open(path, 'rb') as f:
    x = read_sn3_pascalvincent_tensor(f, strict=False)
  assert (x.dtype == torch.uint8)
  assert (x.ndimension() == 3)
  return x

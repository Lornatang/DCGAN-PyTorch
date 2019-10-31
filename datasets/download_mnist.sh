#!/usr/bin/env bash

# create raw folder
mkdir -p MNIST/raw/
# download MNIST train datasets
wget -P MNIST/raw/ http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget -P MNIST/raw/ http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
# download MNIST test datasets
wget -P MNIST/raw/ http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget -P MNIST/raw/ http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

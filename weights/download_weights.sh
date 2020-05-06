#!/bin/bash

FILE=$1

if [[ ${FILE} != "mnist" && ${FILE} != "cifar" && ${FILE} != "imagenet" ]]; then
    echo "Available datasets are: mnist, cifar, imagenet"
    exit 1
fi

URL=https://github.com/Lornatang/DCGAN-PyTorch/releases/download/1.0/${FILE}.zip
ZIP_FILE=${FILE}.zip
TARGET_DIR=${FILE}
wget -N ${URL} -O ${ZIP_FILE}
unzip ${ZIP_FILE}
rm ${ZIP_FILE}

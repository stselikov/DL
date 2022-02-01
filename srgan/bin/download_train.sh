#!/bin/bash

# Download hi-res training dataset
echo "Download hi-res training dataset"

wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip DIV2K_train_HR.zip
mkdir DIV2K_train_HR/1
mv DIV2K_train_HR/*.png  DIV2K_train_HR/1/*
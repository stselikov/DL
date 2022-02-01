#!/bin/bash

# Download low-res validation dataset
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip
unzip DIV2K_valid_LR_bicubic_X4.zip

# Download high-res validation dataset
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
unzip DIV2K_valid_HR.zip 
mkdir test_images
mkdir result_images 
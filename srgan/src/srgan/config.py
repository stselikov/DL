import torch
from torch import nn
from torch import optim
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg19
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt


LOAD_DISC = True    # Define if we load previously saved Discriminator checkpoint
LOAD_GEN = True     # Define if we load previously saved Generator checkpoint
SAVE_MODEL = True   # Define if we save both Discriminator and Generator checkpoints after training
DISC_TRAIN = True   # Define if we train full GAN or just pre-train Generator

WORK_PATH = "."
IMAGE_PATH = "DIV2K_train_HR/1"     # Path to training dataset
CHECKPOINT_GEN = "gen_srgan.pth"    # Generator checkpoint name
CHECKPOINT_DISC = "disc_srgan.pth"  # Discriminator checkpoint name
TEST_IMAGES = "test_images"         # Path to validation images
RESULT_IMAGES = "result_images"     # Path to generated images 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100                    # Number of training epochs
BATCH_SIZE = 16     
NUM_WORKERS = 0                     # Number of workers for Dataloader
HIGH_RES = 96                       # Size of random cropped sub-images of distinct training images
LOW_RES = HIGH_RES // 4             # Size of downsampled sub-images used for training
DISK_CONV = HIGH_RES // (2**4)      # Size of last convolution used in Discriminator model
IMG_CHANNELS = 3                    # Number of image channels
PLOT_FFREQUENCY = 10                # Frequency of validation images generation 
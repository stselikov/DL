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


LOAD_DISC = True
LOAD_GEN = True
SAVE_MODEL = True
DISC_TRAIN = True

WORK_PATH = "."
IMAGE_PATH = "DIV2K_train_HR/1"
CHECKPOINT_GEN = "gen_srgan.pth"
CHECKPOINT_DISC = "disc_srgan.pth"
TEST_IMAGES = "test_images"
RESULT_IMAGES = "result_images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-5
NUM_EPOCHS = 4 
BATCH_SIZE = 16
NUM_WORKERS = 0
HIGH_RES = 96
LOW_RES = HIGH_RES // 4
DISK_CONV = HIGH_RES // (2**4)
IMG_CHANNELS = 3
PLOT_FFREQUENCY = 2
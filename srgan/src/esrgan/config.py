import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg19
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import torchvision.transforms.functional as TF
import random
import gc


LOAD_DISC = True
LOAD_GEN = True

SAVE_MODEL = True

DISC_TRAIN = True # Define if we train full GAN or just pre-train Generator

WORK_PATH = "."
IMAGE_PATH = "DIV2K_train_HR"
CHECKPOINT_GEN = "gen_esrgan.pth"
CHECKPOINT_DISC = "disc_esrgan.pth"
TEST_IMAGES = "test_images"
RESULT_IMAGES = "result_images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 4
BATCH_SIZE = 16
NUM_WORKERS = 8
HIGH_RES = 128
LOW_RES = HIGH_RES // 4
DISK_CONV = HIGH_RES // (2**4)
IMG_CHANNELS = 3
PLOT_FREQUENCY = 2

# Loss function weight initialization
pixel_weight = 1e-2 
content_weight = 1.0
adversarial_weight = 5e-3
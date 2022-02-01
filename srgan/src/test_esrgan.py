# Enchanced Super Resolution GAN test procedures

import torch
from PIL import Image
import os
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as tt
from esrgan.model import Generator


MODEL_PATH = "."
HR_PATH = "DIV2K_valid_HR"

CHECKPOINT_GEN = "gen_esrgan.pth"
LR_PATH = "test_images"
RESULT_IMAGES = "result_images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Test image transformation
test_transform = tt.Compose(
    [
        tt.ToTensor(),
        tt.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    ]
)

# Load pre-trained model state
def load_model(filename, model):
    filename = os.path.join(MODEL_PATH, filename)
    print("=> Loading model from " + filename)
    checkpoint = torch.load(filename, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])


# Reading low-res images and convert them into high-res images 
def plot_examples(gen):
    # For all files in low-res folder
    files = os.listdir(LR_PATH)

    gen.eval()
    loop = tqdm(files, leave=True)
    for file in loop:
        image = Image.open(os.path.join(LR_PATH, file))
        
        with torch.no_grad():
            upscaled_img = gen(test_transform(image).unsqueeze(0).to(DEVICE))
            upscaled_img = upscaled_img.detach().to('cpu')
        # Save reconstructed image to disk
        fname, fext = os.path.splitext(file)
        fname += 'u4'
        save_image(upscaled_img, os.path.join(RESULT_IMAGES, fname + fext))
        loop.set_postfix(file=file)

    return len(files)


def main():
    
    # Loading the pre-trained image generator
    gen = Generator(in_channels=3).to(DEVICE)
    load_model(CHECKPOINT_GEN, gen)

    total_files = plot_examples(gen)
    print(f'Upscaled {total_files} files')


if __name__ == "__main__":
    main()
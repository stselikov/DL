# Super Resolution GAN validation procedures

import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
from srgan.model import Generator


MODEL_PATH = "."
LR_PATH = "DIV2K_valid_LR_bicubic/X4"
HR_PATH = "DIV2K_valid_HR"

CHECKPOINT_GEN = "src/srgan/gen_srgan.pth"
#LR_PATH = "test_images"
RESULT_IMAGES = "result_images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Test image transformation
test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

# PSNR loss function
def psnr_loss(upscaled_img, valid_img):
        psnr = 10. * torch.log10(1. / torch.mean((upscaled_img - valid_img) ** 2))
        return psnr.item()


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
    total_psnr = []
    total_files = []
    gen.eval()
    loop = tqdm(files, leave=True)
    for file in loop:
        psnr = 0.
        image = Image.open(os.path.join(LR_PATH, file))
        
        with torch.no_grad():
            upscaled_img = gen(test_transform(image=np.asarray(image))["image"].unsqueeze(0).to(DEVICE))
            upscaled_img = upscaled_img.detach().to('cpu') * 0.5 + 0.5
        # Save reconstructed image to disk
        fname, fext = os.path.splitext(file)
        fname = fname.replace('x4', 'u4')
        save_image(upscaled_img, os.path.join(RESULT_IMAGES, fname + fext))

        # Trying to find validation hi-res image for an upscaled low-res image
        fname = fname.replace('u4', '')
        valid_file = os.path.join(HR_PATH, fname + fext)
        # Calculating PSNR for validation and reconstructed images
        if os.path.exists(valid_file):
            image = Image.open(valid_file)
            valid_img = test_transform(image=np.asarray(image))["image"]
            psnr = psnr_loss(upscaled_img.squeeze(0), valid_img)
        loop.set_postfix(file=file, psnr=psnr)
        total_psnr.append(psnr)
        total_files.append(file)
    return total_files, total_psnr


def main():
    
    # Loading the pre-trained image generator
    gen = Generator(in_channels=3).to(DEVICE)
    load_model(CHECKPOINT_GEN, gen)

    total_files, total_psnr = plot_examples(gen)
    print('PSNR per files')
    # Processing low-res images
    for i in range(len(total_files)):
        print(f'File name = {total_files[i]}, PSNR = {total_psnr[i]:.2f}')
    
    mean_psnr = np.array(total_psnr).mean()
    print(f'Mean PSNR for all files = {mean_psnr:.4f}')
    # Show PSNR
    plt.figure(figsize=(15, 6))
    plt.bar(range(len(np.array(total_psnr))), np.array(total_psnr))
    plt.xlabel('file number')
    plt.ylabel('psnr')
    plt.title('PSNR metric for files')
    plt.show()
    

if __name__ == "__main__":
    main()
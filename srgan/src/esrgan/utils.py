from esrgan.config import *
from esrgan.dataload import test_transform

# VGG19-based loss module as described in ESRGAN paper
class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:35].eval().to(DEVICE)
        self.loss = nn.L1Loss()

        for param in self.vgg.parameters():
            param.requires_grad = False

        # The preprocessing method of the input data. This is the VGG model preprocessing method of the ImageNet dataset.
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


    def forward(self, input, target):

        # Standardized operations
        input = input.sub(self.mean).div(self.std)
        target = target.sub(self.mean).div(self.std)

        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)

# Saving model checkpoint
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth"):
    filename = os.path.join(WORK_PATH, filename)
    print("=> Saving checkpoint to " + filename)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


# Loading model checkpoint
def load_checkpoint(filename, model, optimizer, lr):
    filename = os.path.join(WORK_PATH, filename)
    print("=> Loading checkpoint from " + filename)
    checkpoint = torch.load(filename, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# Generating a set of hi-res images from low-res validation images
def plot_examples(gen):
    files_folder = os.path.join(WORK_PATH, TEST_IMAGES)
    files = os.listdir(files_folder)

    gen.eval()
    for file in files:
        image = Image.open(os.path.join(files_folder, file))
        with torch.no_grad():
            upscaled_img = gen(test_transform(image).unsqueeze(0).to(DEVICE))
        save_image(upscaled_img, os.path.join(os.path.join(WORK_PATH, RESULT_IMAGES), file))
    gc.collect()
    torch.cuda.empty_cache()
    gen.train()

from srgan.config import *
# High-res images transformations
highres_transform = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ]
)

# Low-res images transformations
lowres_transform = A.Compose(
    [
        A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

# High-res and lowres initial images transformations
both_transforms = A.Compose(
    [
        A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
    ]
)

# Test images transformations
test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

# Simple class for loading images from folder
class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)    

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_file = self.files[index]
        image = np.array(Image.open(os.path.join(self.root_dir, img_file)))
        image = both_transforms(image=image)["image"]
        high_res = highres_transform(image=image)["image"]
        low_res = lowres_transform(image=image)["image"]
        return low_res, high_res
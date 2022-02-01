from esrgan.config import *

# High-res images transformations
highres_transform = tt.Compose(
    [
        tt.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    ]
)

# Low-res images transformations
lowres_transform = tt.Compose(
    [
        tt.Resize(size=[LOW_RES, LOW_RES], interpolation=tt.InterpolationMode.BICUBIC),
        tt.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    ]
)

# High-res and lowres initial images transformations
both_transforms = tt.Compose(
    [   
        tt.RandomCrop(size=[HIGH_RES, HIGH_RES]),
        tt.RandomHorizontalFlip(p=0.5),
        tt.RandomRotation(degrees=(-90, 90), interpolation=tt.InterpolationMode.BICUBIC),
        tt.ToTensor(),
    ]
)

# Test images transformations
test_transform = tt.Compose(
    [
        tt.ToTensor(),
        tt.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    ]
)
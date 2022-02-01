# SRGAN  model
import torch
from torch import nn
from srgan.config import DISK_CONV

# Residual block
class Residual_Block(nn.Module):
    def __init__(self, num_channels):
        super(Residual_Block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.PReLU(num_parameters=num_channels),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channels)
        )

    def forward(self, x):
        out = self.conv(x)
        out = torch.add(out, x)
        return out


# Generator module
class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16):
        super(Generator, self).__init__()

        self.conv_initial = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=9, stride=1, padding=4, bias=True),
            nn.PReLU(num_parameters=num_channels)
        )

        self.residuals_sequence = nn.Sequential(*[Residual_Block(num_channels) for _ in range(num_blocks)])
        
        self.conv_midddle = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channels)
        )
        
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels * 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.PReLU(num_parameters=num_channels),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels * 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.PReLU(num_parameters=num_channels)
        )

        self.conv_final = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4, bias=True)


    def forward(self, x):
        initial = self.conv_initial(x)
        out = self.residuals_sequence(initial)
        out = self.conv_midddle(out)
        out = torch.add(out, initial)
        out = self.upsample(out)
        out = self.conv_final(out)
        out = torch.tanh(out)
        return out


# Convolution block
class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        # (3) x 96 x 96
        self.conv_initial = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # (64) x 96 x 96

        self.conv_sequence = nn.Sequential(
            Conv_Block(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False), # (64) x 48 x 48     
            Conv_Block(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False), # (128) x 48 x 48           
            Conv_Block(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False), # (128) x 24 x 24
            Conv_Block(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False), # (256) x 24 x 24
            Conv_Block(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False), # (256) x 12 x 12
            Conv_Block(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),  # (512) x 12 x 12          
            Conv_Block(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False),  # (512) x 6 x 6          
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*DISK_CONV*DISK_CONV, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        out = self.conv_initial(x)
        out = self.conv_sequence(out)
        out = self.classifier(out)
        return out
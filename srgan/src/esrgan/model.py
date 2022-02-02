# ESRGAN  model
import torch
from torch import nn
import torch.nn.functional as F
from esrgan.config import DISK_CONV

# Simple Convolution block without Batch Normalization
class Conv_Block_No_BN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv_Block_No_BN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


# Residual Dense block
class Residual_Dense_Block(nn.Module):
    def __init__(self, in_channels=64, channels=32, residual_beta=0.2):
        super(Residual_Dense_Block, self).__init__()
        self.residual_beta = residual_beta
        
        self.conv1 = Conv_Block_No_BN(in_channels = in_channels, 
                                        out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv_Block_No_BN(in_channels = in_channels + channels, 
                                        out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = Conv_Block_No_BN(in_channels = in_channels + 2 * channels, 
                                        out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = Conv_Block_No_BN(in_channels = in_channels + 3 * channels, 
                                        out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels = in_channels + 4 * channels, 
                                        out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), dim=1))
        x3 = self.conv3(torch.cat((x, x1, x2), dim=1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), dim=1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x5 * self.residual_beta + x


# Residual in Residual Dense block (RRDB)
class RRDB(nn.Module):
    def __init__(self, in_channels=64, channels=32, residual_beta=0.2):
        super(RRDB, self).__init__()
        self.residual_beta = residual_beta
        
        self.rrdb = nn.Sequential(
            Residual_Dense_Block(in_channels, channels),
            Residual_Dense_Block(in_channels, channels),
            Residual_Dense_Block(in_channels, channels)
        )

    def forward(self, x):
        return self.rrdb(x) * self.residual_beta + x


# Simple Upsample block
class Upsample_Block(nn.Module):
    def __init__(self, channels, scale_factor=2):
        super(Upsample_Block, self).__init__()
        self.scale_factor = scale_factor
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        out =self.upsample(F.interpolate(x, scale_factor=self.scale_factor, mode='nearest'))
        return out

# Generator model
class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=23):
        super(Generator, self).__init__()

        self.conv_initial = nn.Conv2d(in_channels=in_channels, 
                                    out_channels=num_channels, kernel_size=3, stride=1, padding=1, bias=True)


        self.residuals_sequence = nn.Sequential(*[RRDB(num_channels) for _ in range(num_blocks)])

        self.conv_middle = nn.Conv2d(in_channels=num_channels, 
                                    out_channels=num_channels, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.upsample = nn.Sequential(
            Upsample_Block(num_channels), 
            Upsample_Block(num_channels)
        )
        
        self.conv_final = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=num_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        )

    def forward(self, x):
        initial = self.conv_initial(x)
        out = self.residuals_sequence(initial)
        out = self.conv_middle(out)
        out = torch.add(out, initial)
        out = self.upsample(out)
        out = self.conv_final(out)
        return out


# Simple Convolution block
class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


# Discrimonator model
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        # (3) x 128 x 128
        self.conv_initial = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        # (64) x 128 x 128

        self.conv_sequence = nn.Sequential(
            Conv_Block(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False), # (64) x 64 x 64
            Conv_Block(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False), # (128) x 64 x 64
            Conv_Block(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False), # (128) x 32 x 32
            Conv_Block(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False), # (256) x 32 x 32
            Conv_Block(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False), # (256) x 16 x 16
            Conv_Block(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),  # (512) x 16 x 16
            Conv_Block(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)  # (512) x 8 x 8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*DISK_CONV*DISK_CONV, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        out = self.conv_initial(x)
        out = self.conv_sequence(out)
        out = self.classifier(out)
        return out


# Initial model weight initialization (as described in ESRGAN paper)
def initialize_weights(model, scale=0.1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale
            if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale
            if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
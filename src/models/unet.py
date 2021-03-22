import torch
from torch import nn
import torch.nn.functional as F

def downsample_block(end_size, first_layer=False):
    first_size = 3 if first_layer else end_size//2
    pad = 1
    layers = [
        nn.Conv2d(first_size, end_size, 3, padding=pad),
        nn.BatchNorm2d(end_size),
        nn.ReLU(inplace=True),
        nn.Conv2d(end_size, end_size, 3, padding=pad),
        nn.BatchNorm2d(end_size),
        nn.ReLU(inplace=True),
        ]
    if not first_layer:
        layers.insert(0, nn.MaxPool2d(2, stride=2))
    return nn.Sequential(*layers)

def upsample_block(start_size):
    pad = 1
    return nn.Sequential(
        # Upsample Block n, part 2
        nn.Conv2d(start_size, start_size//2, 3, padding=pad),
        nn.BatchNorm2d(start_size//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(start_size//2, start_size//2, 3, padding=pad),
        nn.BatchNorm2d(start_size//2),
        nn.ReLU(inplace=True),
        # Upsample Block n+1, part 1
        #  nn.UpsamplingBilinear2d(scale_factor=2),
        nn.ConvTranspose2d(start_size//2, start_size//4, kernel_size=2, stride=2)
    )

def crop(tensor, shape):
    _, _, dh, dw = shape
    _, _, h, w = tensor.shape
    gap_h = (h - dh) // 2
    gap_w = (w - dw) // 2
    return tensor[:, :, gap_h:gap_h+dh, gap_w:gap_w+dw]


class UNet(nn.Module):
    def __init__(self, output_size):
        super(UNet, self).__init__()
        # Contraction side
        bs = 64
        self.net1 = downsample_block(bs, first_layer=True) # first two blue
        self.net2 = downsample_block(2*bs) # red then two blue
        self.net3 = downsample_block(4*bs)
        self.net4 = downsample_block(8*bs)
        self.net5 = downsample_block(16*bs)
        # Expansive side
        self.net6 = nn.ConvTranspose2d(16*bs, 8*bs, kernel_size=2, stride=2) # first green arrow
        self.net7 = upsample_block(16*bs) # two blue, 1 green
        self.net8 = upsample_block(8*bs)
        self.net9 = upsample_block(4*bs)
        pad = 1
        self.net10 = nn.Sequential(
            nn.Conv2d(2*bs, 2*bs, 3, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*bs, 2*bs, 3, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*bs, output_size, 1),
            nn.Softmax2d()
        )
        self.scaling_steps = 0

    def forward(self, data):
        # (B, C, H, W)
        assert(len(data.shape) == 4)
        # These produce the 4 arrows for short cuts
        x1 = self.net1(data)
        x2 = self.net2(x1)
        x3 = self.net3(x2)
        x4 = self.net4(x3)
        # Last red arrow and 2 blue arrows
        x = self.net5(x4)
        x = self.net6(x)
        x = torch.cat((x, crop(x4, x.shape)), dim=1)
        x = self.net7(x)
        x = torch.cat((x, crop(x3, x.shape)), dim=1)
        x = self.net8(x)
        x = torch.cat((x, crop(x2, x.shape)), dim=1)
        x = self.net9(x)
        x = torch.cat((x, crop(x1, x.shape)), dim=1)
        x = self.net10(x)
        return x

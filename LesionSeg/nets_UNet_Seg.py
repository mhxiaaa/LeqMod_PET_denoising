import torch
import torch.nn as nn

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)

class ResDoubleConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(channels, channels, kernel_size=3, padding=1),
                                   nn.InstanceNorm3d(channels), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(channels, channels, kernel_size=3, padding=1),
                                   nn.InstanceNorm3d(channels), nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1 + x)
        return x2

class UNet_enc(nn.Module):
    def __init__(self, n_channels=1, c=[32, 64, 128, 256]):
        super(UNet_enc, self).__init__()
        self.inc = nn.Sequential(nn.Conv3d(n_channels, c[0], kernel_size=7, padding=3),
                                 ResDoubleConv(c[0])) # B,32,80,80,80
        self.arm1 = nn.Sequential(nn.Conv3d(c[0], c[1], kernel_size=3, stride=2, padding=1),
                                  ResDoubleConv(c[1])) # B,64,40,40,40
        self.arm2 = nn.Sequential(nn.Conv3d(c[1], c[2], kernel_size=3, stride=2, padding=1),
                                  ResDoubleConv(c[2])) # B,128,20,20,20
        self.arm3 = nn.Sequential(nn.Conv3d(c[2], c[3], kernel_size=3, stride=2, padding=1),
                                  ResDoubleConv(c[3])) # B,256,10,10,10

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.arm1(x0)
        x2 = self.arm2(x1)
        x3 = self.arm3(x2)
        return [x0, x1, x2, x3]

class UNet_dec(nn.Module):
    def __init__(self, n_classes=2, c=[32, 64, 128, 256]):
        super(UNet_dec, self).__init__()
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.arm3 = nn.Sequential(nn.Conv3d(c[3]+c[2], c[2], kernel_size=3, padding=1),
                                  nn.InstanceNorm3d(c[2]), nn.ReLU(inplace=True))
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.arm2 = nn.Sequential(nn.Conv3d(c[2]+c[1], c[1], kernel_size=3, padding=1),
                                  nn.InstanceNorm3d(c[1]), nn.ReLU(inplace=True))
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.arm1 = nn.Sequential(nn.Conv3d(c[1]+c[0], c[0], kernel_size=3, padding=1),
                                  nn.InstanceNorm3d(c[0]), nn.ReLU(inplace=True))
        
        self.out3 = nn.Conv3d(c[3], n_classes, kernel_size=1)
        self.out2 = nn.Conv3d(c[2], n_classes, kernel_size=1)
        self.out1 = nn.Conv3d(c[1], n_classes, kernel_size=1)
        self.out0 = nn.Conv3d(c[0], n_classes, kernel_size=1)

    def forward(self, fea):
        [x0, x1, x2, x3] = fea
        x2_dec = self.arm3(torch.cat((self.up3(x3), x2), dim=1))
        x1_dec = self.arm2(torch.cat((self.up2(x2_dec), x1), dim=1))
        x0_dec = self.arm1(torch.cat((self.up1(x1_dec), x0), dim=1))
        out3 = self.out3(x3)
        out2 = self.out2(x2_dec)
        out1 = self.out1(x1_dec)
        out0 = self.out0(x0_dec)
        return out0, out1, out2, out3

class Unet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, c=[32, 64, 128, 256]):
        super(Unet, self).__init__()
        self.enc = UNet_enc(n_channels=n_channels, c=c)
        self.dec = UNet_dec(n_classes=n_classes, c=c)

    def forward(self, x):
        [x0, x1, x2, x3] = self.enc(x)
        out0, out1, out2, out3 = self.dec([x0, x1, x2, x3])
        return out0, out1, out2, out3

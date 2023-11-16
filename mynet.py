import torch
import torch.nn as nn
import torch.nn.functional as F


class Down(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out=self.conv(x)
        return out


class Up(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2),
        )
        self.out=nn.Sequential(
            nn.Conv2d(2*out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x,y):
        x = self.up(x)
        diffY = y.size()[2] - x.size()[2]
        diffX = y.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
        out=torch.cat([x,y],dim=1)

        return self.out(out)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        k=64
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, k, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(k),
            nn.ReLU(inplace=True),
            nn.Conv2d(k, k, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(k),
            nn.ReLU(inplace=True),
        )
        self.down1 = Down(k, 2*k)
        self.down2 = Down(2*k, 4*k)
        self.down3 = Down(4*k, 8*k)
        self.down4 = Down(8 * k,16 * k)

        self.up1 = Up(16 * k, 8 * k)
        self.up2 = Up(8*k, 4*k)
        self.up3 = Up(4*k, 2*k)
        self.up4 = Up(2*k, k)

        self.out=nn.Sequential(
            nn.Conv2d(k, n_classes, kernel_size=1),
        )

    def forward(self, x):
        x0=self.inc(x)

        x1=self.down1(x0)
        x2=self.down2(x1)
        x3 = self.down3(x2)
        y = self.down4(x3)

        y=self.up1(y,x3)
        y = self.up2(y,x2)
        y = self.up3(y,x1)
        y = self.up4(y, x0)

        return self.out(y)


if __name__ == '__main__':
    model = UNet(3,1)
    x=torch.randn((1,3,256,256))
    y=model(x)
    print(y.shape)
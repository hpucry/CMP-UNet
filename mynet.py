import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ACA(nn.Module):
    def __init__(self,in_channels,p):
        super().__init__()
        self.size =int(math.log2(in_channels))
        self.avgpool=nn.AdaptiveAvgPool2d(self.size)
        self.maxpool = nn.AdaptiveMaxPool2d(self.size)

        self.q = nn.Sequential(
            nn.Linear(2*self.size**2,(self.size**2)//2),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear((self.size**2)//2, 1),
        )
        self.k = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Conv2d(in_channels//4, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b,c,w,h=x.shape
        max_x=self.maxpool(x).view(b,c,self.size**2)
        avg_x=self.avgpool(x).view(b,c,self.size**2)
        q=torch.cat([max_x,avg_x],dim=2)
        q=self.q(q).view(b,c,1,1)
        k=self.k(q)
        return k*x+x


class MSCAF(nn.Module):
    def __init__(self,in_channels,out_channels,p):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=3,dilation=3,padding=3,bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=3,dilation=5,padding=5,bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Sequential(
            ACA(out_channels,p),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
        )

    def forward(self, x):
        x1=self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        out=torch.cat([x1,x2,x3],dim=1)
        return self.out(out)


class WideConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        if in_channels>out_channels:
            k=out_channels
        else:
            k=in_channels
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=7//2,groups=k,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class ThinConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,1), padding=(3//2,0),bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1,3), padding=(0,3//2),bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x=self.model(x)
        return x


class CFFA(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.thinconv=ThinConv(in_channels,out_channels)
        self.wideconv=WideConv(in_channels,out_channels)
        self.out=nn.Sequential(
            nn.Conv2d(2*out_channels, out_channels, kernel_size=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        thin=self.thinconv(x)
        wide=self.wideconv(x)
        out=torch.cat([thin,wide],dim=1)
        return self.out(out)


class Down(nn.Module):
    def __init__(self,in_channels,out_channels,p):
        super().__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            CFFA(in_channels,out_channels),
            nn.Dropout(p),
            CFFA(out_channels, out_channels),
            nn.Dropout(p),
        )

    def forward(self, x):
        out=self.conv(x)
        return out


class Up(nn.Module):
    def __init__(self,in_channels,out_channels,p):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2),
            nn.Dropout(p),
        )
        self.out=nn.Sequential(
            MSCAF(2*out_channels,out_channels,p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
        )

    def forward(self, x,y):
        x = self.up(x)
        diffY = y.size()[2] - x.size()[2]
        diffX = y.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
        out=torch.cat([x,y],dim=1)
        return self.out(out)


class CMP_UNet(nn.Module):
    def __init__(self, n_channels, n_classes,p=0.2):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        k=32
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, k, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(k),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Conv2d(k, k, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(k),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
        )
        self.down1 = Down(k, 2*k,p)
        self.down2 = Down(2*k, 4*k,p)
        self.down3 = Down(4*k, 8*k,p)
        self.down4 = Down(8 * k,16 * k,p)

        self.up1 = Up(16 * k, 8 * k,p)
        self.up2 = Up(8*k, 4*k,p)
        self.up3 = Up(4*k, 2*k,p)
        self.up4 = Up(2*k, k,p)

        self.out1=nn.Sequential(
            nn.Conv2d(8*k, k//8, kernel_size=1,bias=False),
            nn.BatchNorm2d(k//8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
            nn.Dropout(p),
        )
        self.out2 = nn.Sequential(
            nn.Conv2d(4*k, k//8, kernel_size=1,bias=False),
            nn.BatchNorm2d(k // 8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Dropout(p),
        )
        self.out3 = nn.Sequential(
            nn.Conv2d(2*k, k//4, kernel_size=1,bias=False),
            nn.BatchNorm2d(k // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Dropout(p),
        )
        self.out4 = nn.Sequential(
            nn.Conv2d(k, k//2, kernel_size=1,bias=False),
            nn.BatchNorm2d(k // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
        )
        self.ffm=MSCAF(k,k,p)
        self.out = nn.Sequential(
            nn.Conv2d(k, n_classes, kernel_size=1),
        )

    def padding(self,x,y):
        diffY = y.size()[2] - x.size()[2]
        diffX = y.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x

    def forward(self, x):
        x0=self.inc(x)

        x1=self.down1(x0)
        x2=self.down2(x1)
        x3 = self.down3(x2)
        y = self.down4(x3)

        y1=self.up1(y,x3)
        y2 = self.up2(y1,x2)
        y3 = self.up3(y2,x1)
        y4 = self.up4(y3, x0)

        o1=self.out1(y1)
        o2 = self.out2(y2)
        o3 = self.out3(y3)
        o4 = self.out4(y4)

        o1=self.padding(o1,o4)
        o2 = self.padding(o2, o4)
        o3 = self.padding(o3, o4)

        o=torch.cat([o1,o2,o3,o4],dim=1)
        o=self.ffm(o)+y4

        return self.out(o)

if __name__ == '__main__':
    model = CMP_UNet(3,1)
    x=torch.randn((1,3,256,256))
    y=model(x)
    print(y.shape)

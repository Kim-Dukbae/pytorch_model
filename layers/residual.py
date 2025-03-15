import torch.nn as nn

# Resnet18/34에서 활용되는 layers
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2,padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.downsample(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, downsample= False):
        super(ResidualBlock, self).__init__()
        self.downsample = None
        stride = 2 if downsample ==  True  else 1
        out_channels = in_channels if stride == 1 else in_channels * 2

        self.conv = ConvBlock( in_channels, out_channels ,
                 kernel_size= 3, stride= stride, padding= 1, bias= False,
                 count= 1)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size= 3, stride= 1, padding= 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if downsample:
            self.downsample = Downsample(in_channels, out_channels)
            
    def forward(self, x):
        residual = self.bn2(self.conv2(self.conv(x) ))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + residual)

class Bottleneck(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels, init=False):
        super(Bottleneck, self).__init__()
        
        stride = 2 if init else 1
        
        # 1x1 conv: channel reduction
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, 
                             stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        
        # 3x3 conv
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, 
                             stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        
        # 1x1 conv: channel expansion
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, 
                             stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Downsample residual connection 
        if stride != 1 or in_channels != out_channels:
            self.downsample = Downsample(in_channels, out_channels)
        else:
            self.downsample = nn.Identity()
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual =   self.bn3(self.conv3(self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))))
        x = self.downsample(x)

        return self.relu( residual + x )

class Residual_Block(nn.Module):
    def __init__(self, in_channels, stride=1, downsample=False):
        super(Residual_Block, self).__init__()
        out_channels = in_channels if stride == 1 else in_channels * 2
        self.residual = downsample or stride != 1  # stride=2면 무조건 다운샘플링

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if self.residual:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        
        if self.residual:
            x = self.downsample(x)
        
        return self.relu(residual + x)

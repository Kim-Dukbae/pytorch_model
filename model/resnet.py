'''
ResNet18
ResNet34
ResNet50
ResNet101
ResNet151
'''

class ResNet18(nn.Module):
    def __init__(self, in_channels= 3, out_channels= 64, img_size= 224, hidden_channels= 512, num_classes=1000):
        super(ResNet18, self).__init__()  
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=7, stride=2, padding=3, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # 활성화 함수로 ReLU 추가
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
        )
        self.layer2 = nn.Sequential(
            Residual_Block(out_channels, stride=1, downsample= False),
            Residual_Block(out_channels, stride=1, downsample= False)
        )
        self.layer3 = nn.Sequential(
            Residual_Block(out_channels, stride=2, downsample= True),
            Residual_Block(out_channels*2, stride=1, downsample= False)
        )
        self.layer4 = nn.Sequential(
            Residual_Block(out_channels*2, stride=2, downsample= True),
            Residual_Block(out_channels*4, stride=1, downsample= False)
        )
        self.layer5 = nn.Sequential(
            Residual_Block(out_channels*4, stride=2, downsample= True),
            Residual_Block(out_channels*8, stride=1, downsample= False)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features= out_channels*8, out_features= num_classes)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = nn.Flatten()(x)
        return self.fc(x)

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size= 3, stride= 1, padding= 1, bias= False,
                 count= 2):
        super(ConvBlock, self).__init__()  
        layers = []
        for _ in range(count):
            layers.append(nn.Conv2d(
                in_channels=in_channels if _ == 0 else out_channels,
                out_channels=out_channels,
                kernel_size= kernel_size,
                stride= stride,
                padding= padding,
                bias = bias
            ))
            if bias == False:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Fully_Connected(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(Fully_Connected, self).__init__()
        self.fc1 = nn.Linear(in_features= in_channels, out_features= hidden_channels)
        self.fc2 = nn.Linear(in_features= hidden_channels, out_features= num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2( self.relu( self.fc1(x) ))

class LeNet1(nn.Module):
    def __init__(self):
        super(LeNet1, self).__init__()  
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels= 4, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels= 4, out_channels= 12, kernel_size=5)

        self.avg_pool =  nn.AvgPool2d(kernel_size= 2)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(in_features= 4*4*12, out_features= 10)

    def forward(self, x):
        x = self.avg_pool(self.tanh(self.conv1(x)))
        x = self.avg_pool(self.tanh(self.conv2(x)))
        x = nn.Flatten()(x)
        return self.fc1(x)

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()  
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels= 6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels= 6, out_channels= 16, kernel_size=5)

        self.avg_pool =  nn.AvgPool2d(kernel_size= 2)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(in_features= 5*5*16, out_features= 64)
        self.fc2 = nn.Linear(in_features= 64, out_features= 10)

    def forward(self, x):
        x = self.avg_pool(self.tanh(self.conv1(x)))
        x = self.avg_pool(self.tanh(self.conv2(x)))
        x = nn.Flatten()(x)
        x = self.tanh(self.fc1(x))
        return self.fc2(x)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()  
        self.conv1 = ConvBlock(in_channels=1, out_channels= 6,
                               kernel_size= 3, stride=1, padding= 0,
                               count=2)
        self.conv2 = ConvBlock(in_channels=6, out_channels= 16,
                               kernel_size= 3, stride=1, padding= 0,
                               count=2)
        
        self.max_pool =  nn.MaxPool2d(kernel_size= 2)
        self.fc = Fully_Connected(in_channels= 5*5*16, hidden_channels= 64, 
                                  num_classes= 10)

    def forward(self, x):
        x = self.max_pool(self.conv1(x))
        x = self.max_pool(self.conv2(x))
        x = nn.Flatten()(x)
        return self.fc(x)

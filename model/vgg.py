import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, in_channels, img_size= 224, out_channels= 64, hidden_channels= 4096, num_classes= 1000):
        super(VGG16, self).__init__()  
        # x를 5개의 (Conv -> BN -> ReLU) 연산 블록에 통과시킴
        # 각 블록은 (Conv → BN → ReLU) 연산을 2번 반복
        self.conv1 = ConvBlock(in_channels= in_channels, out_channels= out_channels,
                               kernel_size= 3, stride=1, padding= 1,
                               count=2)
        self.conv2 = ConvBlock(in_channels=out_channels, out_channels= out_channels*2,
                               kernel_size= 3, stride=1, padding= 1,
                               count=2)
        self.conv3 = ConvBlock(in_channels=out_channels*2, out_channels= out_channels*4,
                               kernel_size= 3, stride=1, padding= 1,
                               count=2)
        self.conv4 = ConvBlock(in_channels=out_channels*4, out_channels= out_channels*8,
                               kernel_size= 3, stride=1, padding= 1,
                               count=2)
        self.conv5 = ConvBlock(in_channels=out_channels*8, out_channels= out_channels*8,
                               kernel_size= 3, stride=1, padding= 1,
                               count=2)

        # max pooling 연산은 학습 가능한 파라미터가 없으므로 동일한 레이어를 여러 번 재사용 가능
        self.max_pool =  nn.MaxPool2d(kernel_size= 2)
        
        size = img_size//(2**5) #  마지막 conv layer output size 계산
        self.fc = Fully_Connected(in_channels= size*size*out_channels*8, hidden_channels= hidden_channels, 
                                  num_classes= num_classes)

    def forward(self, x):
        x = self.max_pool(self.conv1(x))
        x = self.max_pool(self.conv2(x))
        x = self.max_pool(self.conv3(x))
        x = self.max_pool(self.conv4(x))
        x = self.max_pool(self.conv5(x))
        x = nn.Flatten()(x)
        return self.fc(x)

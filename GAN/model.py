import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, output_channels, hidden_dim, image_size):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.output_channels = output_channels

        # 노이즈 벡터 -> 초기 특징 맵 변환
        self.init_features = hidden_dim * (image_size // 4) * (image_size // 4)
        self.input_layer = nn.Sequential(
            nn.Linear(latent_dim, self.init_features, bias=False),
            nn.BatchNorm1d(self.init_features),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 업샘플링 레이어
        self.upsample_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 출력 레이어
        self.output_layer = nn.Sequential(
            nn.Conv2d(hidden_dim // 4, output_channels, kernel_size=3, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise):
        x = self.input_layer(noise)
        x = x.view(-1, self.hidden_dim, self.image_size // 4, self.image_size // 4)  # 4D 변환
        x = self.upsample_layer(x)
        x = self.output_layer(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=64, image_size=28):
        super(Discriminator, self).__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.image_size = image_size

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 최종 특징 맵 크기 계산 (image_size // 16)
        self.feature_size = hidden_dim * 8 * (image_size // 16) * (image_size // 16)

        self.output_layer = nn.Linear(self.feature_size, 1)

    def forward(self, image):
        # 특징 추출
        features = self.feature_extractor(image)
        
        # 평탄화 후 출력
        features = features.view(features.size(0), -1)
        output = self.output_layer(features)
        return output

  class GAN(nn.Module):
    def __init__(self, latent_dim=100, output_channels=1, hidden_dim=64, image_size=28):
        super(GAN, self).__init__()
        self.generator = Generator(latent_dim, output_channels, hidden_dim, image_size)
        self.discriminator = Discriminator(output_channels, hidden_dim, image_size)

    def forward(self, x):
        # Generator로 가짜 이미지 생성
        fake_images = self.generator(x)

        # Discriminator로 가짜 이미지 판별
        scores = self.discriminator(fake_images)
        return scores

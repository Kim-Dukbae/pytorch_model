import torch.optim as optim
import numpy as np

def train(dataset, params):
    dataloader = DataLoader(dataset, batch_size= params['batch_size'], shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 초기화
    gan = GAN(
        latent_dim= params['latent_dim'],
        output_channels= 1 if len(dataset.data.shape) < 4 else dataset.data.shape[3],
        hidden_dim= params['hidden_dim'],
        image_size= dataset.data.shape[1]
    ).to(device)

    # 손실 함수 및 최적화 함수 정의
    criterion = nn.BCEWithLogitsLoss()  # 이진 크로스 엔트로피 (로짓 포함)
    optim_D = optim.Adam(gan.discriminator.parameters(), lr= params['lr'])  # Discriminator 최적화
    optim_G = optim.Adam(gan.generator.parameters(), lr= params['lr'])      # Generator 최적화
    
    # 진행도 출력 
    history = {
        'discriminator_loss': [],
        'generator_loss' : [],
        'generator_images': []
    }

    for epoch in range(params['epochs']):
        epoch_loss_d = np.zeros(len(dataloader))
        epoch_loss_g = np.zeros(len(dataloader))

        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)

            # 레이블 생성
            real_label, fake_label = real_fake_label(params['batch_size'], device)

            # --- Discriminator 판별자로 업데이트 초기화 ---
            optim_D.zero_grad()

            # 실제 이미지에 대한 Discriminator 손실
            real_scores = gan.discriminator(real_imgs)
            real_loss = criterion(real_scores, real_label)

            # 가짜 이미지 생성 및 Discriminator 손실
            noise = torch.randn(params['batch_size'], params['latent_dim'],).to(device)
            fake_imgs = gan.generator(noise)
            fake_scores = gan.discriminator(fake_imgs.detach())  # Generator 업데이트 방지
            fake_loss = criterion(fake_scores, fake_label)

            # Discriminator 총 손실
            d_loss = (real_loss + fake_loss) * 0.5
            d_loss.backward()
            optim_D.step()

            # --- Generator 학습 업데이트 초기화 ---
            optim_G.zero_grad()

            # Generator 손실 (가짜 이미지를 진짜로 속이도록)
            fake_scores = gan(noise)  # GAN 전체 forward (G → D)
            g_loss = criterion(fake_scores, real_label)  # Generator는 진짜 레이블 목표

            g_loss.backward()
            optim_G.step()

            progress_bar(i + 1, len(dataloader), epoch, length=100)
            epoch_loss_d[i] = d_loss.item()  # 스칼라 값
            epoch_loss_g[i] = g_loss.item()

        # 25개의 이미지 생성 예정
        noise = torch.randn(params['using noise'], params['latent_dim']).to(device)  
        gen_imgs = gan.generator(noise)
        gen_imgs = 0.5 * gen_imgs.to('cpu') + 0.5  # [-1, 1] -> [0, 1]로 정규화
        gen_imgs = gen_imgs.detach()  # 그래디언트 추적 끊기

        # history 저장
        history['generator_images'].append(gen_imgs.numpy())  # NumPy 변환하여 저장
        history['discriminator_loss'].append( np.mean(epoch_loss_d) )
        history['generator_loss'].append( np.mean(epoch_loss_g) )

    return history, gan

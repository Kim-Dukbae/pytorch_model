# GAN

<p align="center">
  <img src="asset/generator_discriminator.png" />
</p>

generator_discriminator.png
## Model summary
```bash

params = {
    "latent_dim": 100,
    "hidden_dim": 64,
    "batch_size": 32,
    "epochs": 10,
    "lr": 0.001,
    'using noise': 25
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gan = GAN(latent_dim= params['latent_dim'],
          output_channels= 1,
          hidden_dim= params['hidden_dim'],
          image_size= 28).to(device)

from torchsummary import summary
summary(gan, (params['latent_dim'],))
```
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                 [-1, 3136]         313,600
       BatchNorm1d-2                 [-1, 3136]           6,272
         LeakyReLU-3                 [-1, 3136]               0
   ConvTranspose2d-4           [-1, 32, 14, 14]          32,768
       BatchNorm2d-5           [-1, 32, 14, 14]              64
         LeakyReLU-6           [-1, 32, 14, 14]               0
   ConvTranspose2d-7           [-1, 16, 28, 28]           8,192
       BatchNorm2d-8           [-1, 16, 28, 28]              32
         LeakyReLU-9           [-1, 16, 28, 28]               0
           Conv2d-10            [-1, 1, 28, 28]             144
             Tanh-11            [-1, 1, 28, 28]               0
        Generator-12            [-1, 1, 28, 28]               0
           Conv2d-13           [-1, 64, 28, 28]             576
      BatchNorm2d-14           [-1, 64, 28, 28]             128
        LeakyReLU-15           [-1, 64, 28, 28]               0
        MaxPool2d-16           [-1, 64, 14, 14]               0
           Conv2d-17          [-1, 128, 14, 14]          73,728
      BatchNorm2d-18          [-1, 128, 14, 14]             256
        LeakyReLU-19          [-1, 128, 14, 14]               0
        MaxPool2d-20            [-1, 128, 7, 7]               0
           Conv2d-21            [-1, 256, 7, 7]         294,912
      BatchNorm2d-22            [-1, 256, 7, 7]             512
        LeakyReLU-23            [-1, 256, 7, 7]               0
        MaxPool2d-24            [-1, 256, 3, 3]               0
           Conv2d-25            [-1, 512, 3, 3]       1,179,648
      BatchNorm2d-26            [-1, 512, 3, 3]           1,024
        LeakyReLU-27            [-1, 512, 3, 3]               0
        MaxPool2d-28            [-1, 512, 1, 1]               0
           Linear-29                    [-1, 1]             513
    Discriminator-30                    [-1, 1]               0
================================================================
Total params: 1,912,369
Trainable params: 1,912,369
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 2.80
Params size (MB): 7.30
Estimated Total Size (MB): 10.10
----------------------------------------------------------------
```
```bash
history, gan = train(dataset= mnist_dataset, 
                     params = params)

plot_training_history(history)
plot_generator(history, params['epochs']-1):

```

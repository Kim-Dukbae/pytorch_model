# GAN


params = {
    "latent_dim": 100,
    "hidden_dim": 64,
    "batch_size": 32,
    "epochs": 10,
    "lr": 0.001,
    'using noise': 25
}

history, gan = train(dataset= mnist_dataset, 
                     params = params)

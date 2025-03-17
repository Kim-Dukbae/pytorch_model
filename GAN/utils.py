import sys

def progress_bar(progress, total, epoch, length=20):
    percent = 100 * (progress / total)
    filled = int(length * progress / total)
    sys.stdout.write(f"\rEpoch {epoch+1}: {percent:.1f}%")
    sys.stdout.flush()


import matplotlib.pyplot as plt

def plot_training_history(history):
    epochs = len(history['discriminator_loss'])  # 총 스텝 수
    x_axis = range(epochs)

    plt.figure(figsize=(10, 5))

    # Discriminator & Generator Loss
    plt.plot(x_axis, history['discriminator_loss'], label="Discriminator Loss", alpha=0.7)
    plt.plot(x_axis, history['generator_loss'], label="Generator Loss", alpha=0.7)

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_generator(history, num_epoch):
    gen_imgs = history['generator_images'][num_epoch]  # [25, C, H, W] 형태
    
    fig, ax = plt.subplots(5, 5, figsize=(10, 10))
    count = 0
    for row in range(5):
        for col in range(5):
            img = gen_imgs[count].transpose(1, 2, 0)
            if gen_imgs[-1] == 1:
                ax[row, col].imshow(img, cmap= 'gray')
            else:
                ax[row, col].imshow(img)
                
            ax[row, col].axis('off')
            count += 1

    plt.show()

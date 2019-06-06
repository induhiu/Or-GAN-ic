import numpy as np
import gan
import nn
import sys

random_dim = 100

if __name__ == '__main__':
    images = np.fromfile('language_L.txt')
    print(images)
    print(images.shape)
    images = images.reshape(int(images.shape[0] / 784), 28, 28)
    print(images.shape)
    sys.exit()
    gen1 = gan.Generator()
    disc1 = gan.Discriminator()
    my_gan = gan.GAN(random_dim, generator=gen1, discriminator=disc1, x_train=images)
    my_images = my_gan.train(epochs=5)
    # images.tofile('language_L.txt')

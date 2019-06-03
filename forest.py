''' A graph implementation of a forest by Ian Nduhiu, Kenny Talarico and
    Dave Perkins. Created on May 31, 2019 '''

import gan
from random import choice

random_dim = 100

def main():
    gancontrol = gan.GAN(random_dim) #control
    gen = gan.Generator()
    disca = gan.Discriminator()
    discb = gan.Discriminator()
    testgans = [gan.GAN(random_dim, generator=gen, discriminator=disca),
                gan.GAN(random_dim, generator=gen, discriminator=discb)]

    epochs = 15

    gancontrol.train(epochs=epochs)

    for i in range(1, epochs+1):
        choice(testgans).train(id=i)


if __name__ == '__main__':
    main()

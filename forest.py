''' A graph implementation of a forest by Ian Nduhiu, Kenny Talarico and
    Dave Perkins. Created on May 31, 2019 '''

import gan
from random import choice

random_dim = 100

def main():
    gancontrol = gan.GAN(random_dim) #control
    gen = gan.Generator()
    testgans = [gan.GAN(random_dim, generator=gen), gan.GAN(random_dim, generator=gen)]

    epochs = 5

    gancontrol.train(epochs=epochs)

    for i in range(1, epochs+1):
        choice(testgans).train(id=i)


if __name__ == '__main__':
    main()

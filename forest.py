''' A graph implementation of a forest by Ian Nduhiu, Kenny Talarico and
    Dave Perkins. Created on May 31, 2019. '''

import gan
import tree
from random import choice

random_dim = 100

class Forest:
    def __init__(self):
        self.dekutree = tree.Tree()
        self.dekutree.brain.train()
        self.trees = [[None, None, None, None, None],
                      [None, None, None, None, None],
                      [None, None, self.dekutree, None, None],
                      [None, None, None, None, None],
                      [None, None, None, None, None]]


def main():
    # gancontrol = gan.GAN(random_dim) #control
    # gen = gan.Generator()
    # disca = gan.Discriminator()
    # discb = gan.Discriminator()
    # testgans = [gan.GAN(random_dim, generator=gen, discriminator=disca),
    #             gan.GAN(random_dim, generator=gen, discriminator=discb)]
    #
    # epochs = 15
    #
    # gancontrol.train(epochs=epochs)
    #
    # for i in range(1, epochs+1):
    #     choice(testgans).train(id=i)

    # tree1 = tree.Tree()
    # tree2 = tree.Tree()
    #
    # gan1 = gan.GAN(random_dim, generator=tree1.generator, discriminator=tree2.discriminator)
    #
    # for _ in range(1):
    #     gan1.train()
    #     tree1.brain.train(plot=False)
    #     tree2.brain.train(plot=False)
    #
    # tree3 = tree.Tree()
    #
    # gan2 = gan.GAN(random_dim, generator=tree3.generator, discriminator=tree1.discriminator)
    #
    # gan2.train()

    gen = gan.Generator()
    disc1 = gan.Discriminator()
    disc2 = gan.Discriminator()

    gan1 = gan.GAN(random_dim, generator=gen, discriminator=disc1)
    gan1.train(epochs=10)

    gan2 = gan.GAN(random_dim, generator=gen, discriminator=disc2)
    gan2.train(epochs=5)



if __name__ == '__main__':
    main()

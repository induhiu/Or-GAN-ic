""" Implementation for a "tree" class. By Kenny Talarico and Ian Nduhiu, June
    2019. """

import gan

random_dim = 100

class Tree:
    def __init__(self):
        self.generator = gan.Generator()
        self.discriminator = gan.Discriminator()
        self.brain = gan.GAN(random_dim, generator=self.generator, discriminator=self.discriminator)

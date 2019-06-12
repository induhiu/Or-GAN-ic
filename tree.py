""" Implementation for a "tree" class. By Kenny Talarico and Ian Nduhiu, June
    2019. """

import gan
import math
from secrets import randbelow

random_dim = 100

class Tree:
    def __init__(self, location, parent=None):
        self.generator = gan.Generator()
        self.discriminator = gan.Discriminator()
        self.age = 0
        self.location = location
        self.parent = parent

    def __str__(self):
        return "TREE, " + str(self.age)

    def spawnChild(self):
        num = randbelow(628) / 100
        return Tree(location=(round(math.cos(num) * self.age * 10, 2), round(math.sin(num) * self.age * 10, 2)), parent=self)

    def resetDiscriminator(self):
        self.discriminator = gan.Discriminator()

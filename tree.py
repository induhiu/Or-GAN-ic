""" Implementation for a "tree" class. By Kenny Talarico and Ian Nduhiu, June
    2019. """

import gan

random_dim = 100

class Tree:
    def __init__(self, id):
        self.generator = gan.Generator()
        self.discriminator = gan.Discriminator()
        self.age = 0
        self.id = id

    def __str__(self):
        return "TREE, " + str(self.age)

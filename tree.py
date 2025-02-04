""" Implementation for a "tree" class. By Kenny Talarico and Ian Nduhiu, June
    2019. """

from gan import Generator
from gan import Discriminator
from gan import GAN
import math
from secrets import randbelow
from language_getter import produce_language
from tensorflow.keras.models import load_model

random_dim = 100

class Tree:
    def __init__(self, location, forest, parent=None, generator=Generator(), name='unnamed'):
        """ Constructor for a Tree. """
        self.generator = generator
        self.discriminator = Discriminator()
        self.nn = load_model('new_nn.h5')
        self.age = 1
        self.location = location
        self.parent = parent
        self.forest = forest
        self.neighbors = ([parent] if parent else [])
        self.name = name

    def __repr__(self):
        """ Overwrite default str representation to output
            T(<position in list of trees>) """
        return 'T(' + str(self.forest.trees.index(self)) + ')'

    def _newlocation(self):
        """ Find a suitable location for a child. """
        num = randbelow(628) / 100
        # A tree's roots are log10 of its age.
        r = math.log10(self.age)
        loc = (self.location[0] + round(math.cos(num) * r, 2), self.location[1] + round(math.sin(num) * r, 2))
        for tree in self.forest.trees:
            if (loc[0]-tree.location[0]) ** 2 + (loc[1]-tree.location[1]) ** 2 <= math.log10(tree.age) ** 2:
                return None
        return loc

    def spawnChild(self):
        """ Attempt to spawn a child. If successful, return it. If not, return None. """
        for _ in range(10):
            loc = self._newlocation()
            if loc:
                break
        if not loc:
            return None
        r = math.log10(self.age)
        child = Tree(location=loc, forest=self.forest, parent=self, name=self.forest.names.pop(0))
        self.forest.connections[child] = [self]
        self.forest.connections[self].append(child)
        self.neighbors.append(child)
        return child

    def resetDiscriminator(self):
        """ Reset this tree's discriminator. """
        self.discriminator.reset()

    def getnewneighbors(self):
        """ Check for neighbors within radius of roots. If found, add all necessary
            connections within both trees and within forest. """
        for t in self.forest.trees:
            if (t is not self.parent and t is not self and t not in self.neighbors and
                ((t.location[0] - self.location[0]) ** 2 + (t.location[1] - self.location[1]) ** 2) <= (1.4 * math.log10(self.age)) ** 2):
                self.forest.connections[self].append(t)
                self.forest.connections[t].append(self)
                self.neighbors.append(t)
                t.neighbors.append(self)

""" Implementation for a "tree" class. By Kenny Talarico and Ian Nduhiu, June
    2019. """

from gan import Generator
from gan import Discriminator
from gan import GAN
import math
from secrets import randbelow

random_dim = 100

class Tree:
    def __init__(self, location, forest, parent=None, generator=Generator()):
        self.generator = generator
        self.discriminator = Discriminator()
        self.age = 1
        self.location = location
        self.parent = parent
        self.forest = forest
        self.neighbors = ([parent] if parent else [])

    def __str__(self):
        return ('DEKU TREE' if not self.parent else 'TREE') + ', age: ' + str(self.age) + ', radius: ' + str(self.age * 10 if self.age <= 15 else 150) + ', location: ' + str(self.location)

    def _newlocation(self):
        num = randbelow(628) / 100
        r = math.log10(self.age)
        # r = 10
        loc = (self.location[0] + round(math.cos(num) * r, 2), self.location[1] + round(math.sin(num) * r, 2))
        for tree in self.forest.trees:
            if (loc[0]-tree.location[0]) ** 2 + (loc[1]-tree.location[1]) ** 2 <= math.log10(tree.age) ** 2:
                return None
        return loc

    def spawnChild(self):
        for _ in range(10):
            loc = self._newlocation()
            if loc:
                break
        if not loc:
            return None
        r = math.log10(self.age)
        child = Tree(location=loc, forest=self.forest, parent=self)
        self.forest.connections[child] = [self]
        self.forest.connections[self].append(child)
        self.neighbors.append(child)
        return child

    def resetDiscriminator(self):
        self.discriminator = Discriminator()

    def getnewneighbors(self):
        for t in self.forest.trees:
            if (t is not self.parent and t is not self and t not in self.neighbors and
                # ((t.location[0] - self.location[0]) ** 2 + (t.location[1] - self.location[1])) <= (2.5 * math.log10(self.age)) ** 2):
                ((t.location[0] - self.location[0]) ** 2 + (t.location[1] - self.location[1]) ** 2) <= (0.7 * math.log10(self.age)) ** 2):
                self.forest.connections[self].append(t)
                self.forest.connections[t].append(self)
                self.neighbors.append(t)
                t.neighbors.append(self)

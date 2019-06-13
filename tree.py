""" Implementation for a "tree" class. By Kenny Talarico and Ian Nduhiu, June
    2019. """

import gan
import math
from secrets import randbelow

random_dim = 100

class Tree:
    def __init__(self, location, forest, parent=None):
        self.generator = gan.Generator()
        self.discriminator = gan.Discriminator()
        self.age = 0
        self.location = location
        self.parent = parent
        self.forest = forest
        self.neighbors = ([parent] if parent else [])

    def __str__(self):
        return ('DEKU TREE' if self.parent is None else 'TREE') + ', age: ' + str(self.age) + ', radius: ' + str(self.age * 10 if self.age <= 15 else 150) + ', location: ' + str(self.location)

    def spawnChild(self):
        num = randbelow(628) / 100
        r = (self.age * 10 if self.age <= 15 else 150)
        child = Tree(location=(round(math.cos(num) * r, 2), round(math.sin(num) * r, 2)),
                     forest=self.forest,
                     parent=self)
        self.forest.connections[child] = [self]
        self.forest.connections[self].append(child)
        self.neighbors.append(child)
        return child

    def resetDiscriminator(self):
        self.discriminator = gan.Discriminator()

    def getnewneighbors(self):
        for t in self.forest.trees:
            if (t is not self.parent and t is not self and t not in self.neighbors and
                ((t.location[0] - self.location[0]) ** 2 + (t.location[1] - self.location[1])) <=  (self.age * 10 if self.age <= 15 else 150) ** 2):
                self.forest.connections[self].append(t)
                self.forest.connections[t].append(self)
                self.neighbors.append(t)
                t.neighbors.append(self)

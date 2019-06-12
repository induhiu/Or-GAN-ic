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

    def __str__(self):
        return ('DEKU TREE' if self.parent is None else 'TREE') + ', age: ' + str(self.age) + ', radius: ' + str(self.age * 10) + ', connections: ' + str(len(self.forest.connections[self.location]))

    def spawnChild(self):
        num = randbelow(628) / 100
        return Tree(location=(round(math.cos(num) * self.age * 10, 2), round(math.sin(num) * self.age * 10, 2)), forest=self.forest, parent=self)

    def resetDiscriminator(self):
        self.discriminator = gan.Discriminator()

    def getnewneighbors(self):
        for t in list(self.forest.locations.values()):
            if (t is not self.parent and t is not self and t.location not in self.forest.connections[self.location] and
                ((t.location[0] - self.location[0]) ** 2 + (t.location[1] - self.location[1])) <=  (self.age * 10) ** 2):
                self.forest.connections[self.location].append(t.location)
                self.forest.connections[t.location].append(self.location)

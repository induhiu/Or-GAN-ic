""" A 'forest' class. Represents a collection of 'trees' designed to communicate
    with one another due to their each having the necessary properties of
    Generative Adversarial Nets (GANs).
    Implementation by Kenny Talarico as part of summer research with Ian Nduhiu
    and advisor David Perkins, June 2019. """

import gan
import tree
from random import choice
import numpy as np
from secrets import randbelow

random_dim = 100
directions = ((1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1))

class Forest:
    def __init__(self, size):
        # a size smaller than 3 would allow for a "deku tree" that is on the
        # edge of the graph, which is undesirable
        assert size >= 3
        # the deku tree, named for the Legend of Zelda character of the same
        # name, is the oldest tree in the forest and exists directly in its
        # center. It is the only tree that knows how to speak "the common
        # language".
        self._dekutree = tree.Tree(id=(size // 2, size // 2))
        # self.trees will be a list of all active trees, and self.forest will
        # be a two dimensional array that mimics the layout of the forest itself.
        # self.forest's dimensions are  size  by  size , and in spaces that don't hold
        # trees the array contains a  None .
        self.trees = [self._dekutree]
        self.forest = [[None for _ in range(size)] for _ in range(size)]
        self.forest[size // 2][size // 2] = self._dekutree

        self.size = size

    def addTreeHere(self, rc):
        """ Add a tree at the coordinates given in tuple  rc  ."""
        assert self.hasNeighbor((rc[0], rc[1]))
        self.forest[rc[0]][rc[1]] = t = tree.Tree(id=rc)
        self.trees.append(t)

    def addTree(self):
        """ Add a tree to the graph. """
        # choose a random space in the forest among those that neighbor an existing
        # tree and don't already contain a tree, and add a sapling there.
        self.addTreeHere(choice([(r, c) for c in range(self.size) for r in range(self.size) if self.hasNeighbor((r, c)) and not self.treeHere((r, c))]))

    def treeHere(self, rc):
        """ Returns True if there is a tree at coordinates rc, False otherwise. """
        return self.forest[rc[0]][rc[1]] is not None

    def hasNeighbor(self, rc):
        """ Returns True if the space given at coordinates rc borders an existing
            tree, False otherwise. """
        assert 0 <= rc[0] < self.size
        assert 0 <= rc[1] < self.size
        for d in directions:
            if 0 <= rc[0] + d[0] < self.size and 0 <= rc[1] + d[1] < self.size and self.treeHere((rc[0] + d[0], rc[1] + d[1])):
                return True
        return False

    def grow(self, rate=1):
        """ The forest ages. All trees get their age incremented, and new
            trees appear at the rate given as a parameter. """
        for t in self.trees:
            t.age += 1
        while randbelow(100) < rate * 100:
            self.addTree(); rate -= 1

    def __str__(self):
        output = ''
        for li in self.forest:
            for t in li:
                output += (('X' + ' ' * 9) if t is None else (t.__str__() + (10-len(t.__str__())) * ' '))
            output += '\n\n'
        return output

    def communicate(self):
        """ Train the generators and discriminators of the relevant trees. """
        for t in self.trees:
            for d in directions:
                if self.forest[t.id[0] + d[0]][t.id[1] + d[1]] and self.forest[t.id[0] + d[0]][t.id[1] + d[1]].age > t.age:
                    gan.GAN(generator=t.generator, discriminator=self.forest[t.id[0] + d[0]][t.id[1] + d[1]].discriminator).train()
                    # this gan needs to be fed the language of its discriminator's tree's generator

def main():
    forest = Forest(5)
    for _ in range(20):
        forest.grow()
    print(forest)

if __name__ == '__main__':
    main()

""" A 'forest' class. Represents a collection of 'trees' designed to communicate
    with one another due to their each having the necessary properties of
    Generative Adversarial Nets (GANs).
    Implementation by Kenny Talarico as part of summer research with Ian Nduhiu
    and advisor David Perkins, June 2019. """

from gan import Generator
from gan import Discriminator
from gan import GAN
import tree
from random import choice
import numpy as np
from secrets import randbelow
from language_getter import produce_language
from graphing import graph as visualize
from tensorflow.keras.models import load_model

random_dim = 100
directions = ((1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1))

# class ForestGrid:
#     def __init__(self, size):
#         # a size smaller than 3 would allow for a "deku tree" that is on the
#         # edge of the graph, which is undesirable
#         assert size >= 3
#         # the deku tree, named for the Legend of Zelda character of the same
#         # name, is the oldest tree in the forest and exists directly in its
#         # center. It is the only tree that knows how to speak "the common
#         # language".
#         self._dekutree = tree.Tree(id=(size // 2, size // 2))
#         # self.trees will be a list of all active trees, and self.forest will
#         # be a two dimensional array that mimics the layout of the forest itself.
#         # self.forest's dimensions are  size  by  size , and in spaces that don't hold
#         # trees the array contains a  None .
#         self.trees = [self._dekutree]
#         self.forest = [[None for _ in range(size)] for _ in range(size)]
#         self.forest[size // 2][size // 2] = self._dekutree
#
#         self.size = size
#
#     def addTreeHere(self, rc):
#         """ Add a tree at the coordinates given in tuple  rc  ."""
#         assert self.hasNeighbor((rc[0], rc[1]))
#         self.forest[rc[0]][rc[1]] = t = tree.Tree(id=rc)
#         self.trees.append(t)
#
#     def addTree(self):
#         """ Add a tree to the graph. """
#         # choose a random space in the forest among those that neighbor an existing
#         # tree and don't already contain a tree, and add a sapling there.
#         self.addTreeHere(choice([(r, c) for c in range(self.size) for r in range(self.size) if self.hasNeighbor((r, c)) and not self.treeHere((r, c))]))
#
#     def treeHere(self, rc):
#         """ Returns True if there is a tree at coordinates rc, False otherwise. """
#         return self.forest[rc[0]][rc[1]] is not None
#
#     def hasNeighbor(self, rc):
#         """ Returns True if the space given at coordinates rc borders an existing
#             tree, False otherwise. """
#         assert 0 <= rc[0] < self.size
#         assert 0 <= rc[1] < self.size
#         for d in directions:
#             if 0 <= rc[0] + d[0] < self.size and 0 <= rc[1] + d[1] < self.size and self.treeHere((rc[0] + d[0], rc[1] + d[1])):
#                 return True
#         return False
#
#     def grow(self, rate=1):
#         """ The forest ages. All trees get their age incremented, and new
#             trees appear at the rate given as a parameter. """
#         print('Growing...')
#         for t in self.trees:
#             t.age += 1
#         while randbelow(100) < rate * 100:
#             self.addTree(); rate -= 1
#
#     def __str__(self):
#         output = ''
#         for li in self.forest:
#             for t in li:
#                 output += (('X' + ' ' * 9) if not t else (t.__str__() + (10-len(t.__str__())) * ' '))
#             output += '\n\n'
#         return output
#
#     def communicate(self):
#         """ Train the generators and discriminators of the relevant trees. """
#         for t in self.trees:
#             for d in directions:
#                 if self.forest[t.id[0] + d[0]][t.id[1] + d[1]] and self.forest[t.id[0] + d[0]][t.id[1] + d[1]].age > t.age:
#                     GAN(generator=t.generator, discriminator=self.forest[t.id[0] + d[0]][t.id[1] + d[1]].discriminator).train()
                    # this gan needs to be fed the language of its discriminator's tree's generator

class Forest:
    def __init__(self):
        print('\n', 'In the vast, deep forest of Hyrule...', '\n', 'Long have I served as the guardian spirit...', '\n', 'I am known as the Deku Tree...', '\n\n', sep='')
        print("Creating Deku Tree...")
        self.deku = tree.Tree(location=(0, 0), forest=self, generator=Generator(g=load_model('./saveddekus/DEKU40.h5')))
        self.trees = [self.deku]
        self.connections = {self.deku: []}
        print("Forest generated!")

    def grow(self, rate=1, years=1):
        for _ in range(years):
            r = rate
            self.age()
            # for t in self.trees:
            #     t.getnewneighbors()
            while randbelow(100) < r * 100:
                self.spawn()
                r -= 1

    def spawn(self):
        parent = choice(self.trees)
        baby = None
        tries = 0
        while not baby:
            while parent.age == 1:
                parent = choice(self.trees)
            baby = parent.spawnChild()
            tries += 1
            if tries == 10:
                return
        self.trees.append(baby)
        #self.trees.append(self.deku.spawnChild())

    def age(self):
        for t in self.trees:
            t.age += 1

    def allParentCommunicate(self):
        for t in self.trees:
            if t.parent: #handle deku
                GAN(generator=t.generator, discriminator=t.parent.discriminator, x_train=produce_language(t.parent.generator)).train()

    def graph(self):
        visualize(self.trees)


def main():
    forest = Forest()
    forest.grow(rate=1, years=10)
    #forest.grow(rate=8, years=2)
    forest.graph()


if __name__ == '__main__':
    main()

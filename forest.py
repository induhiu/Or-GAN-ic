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
            for t in self.trees:
                t.getnewneighbors()
            while randbelow(100) < r * 100:
                self.spawn()
                r -= 1

    def spawn(self):
        baby = None
        while not baby:
            parent = choice(self.trees)
            while parent.age == 1:
                parent = choice(self.trees)
            baby = parent.spawnChild()
        self.trees.append(baby)
        #self.trees.append(self.deku.spawnChild())

    def age(self):
        for t in self.trees:
            t.age += 1

    def graph(self, year='test'):
        visualize(self.trees, str(year))


def main():
    forest = Forest()
    forest.grow(rate=.3, years=50)
    #forest.grow(rate=8, years=2)
    forest.graph()


if __name__ == '__main__':
    main()

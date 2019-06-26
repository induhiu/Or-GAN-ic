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
from pickle import load

random_dim = 100
directions = ((1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1))

class Forest:
    def __init__(self):
        print('\n', 'In the vast, deep forest of Hyrule...', '\n', 'Long have I served as the guardian spirit...', '\n', 'I am known as the Deku Tree...', '\n\n', sep='')
        print("Creating Deku Tree...")
        self.names = load(open('names.txt', 'rb'))
        self.deku = tree.Tree(location=(0, 0), forest=self, generator=Generator(g=load_model('./saveddekus/DEKU60.h5')), name=self.names.pop(0))
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
            # self.communicate()

    def communicate(self):
        # change time variable for how well trained the tree's generator will be
        time = 1
        # [tr.communicate(t) for t in self.connections[tr] for tr in self.trees for _ in range(time)]
        for _ in range(time):
            for tr in self.trees:
                for t in self.connections[tr]:
                    if tr.age < t.age:
                        tr.communicate(t)

        [tree.resetDiscriminator() for tree in self.trees]


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
    forest.grow(rate=1, years=10)
    #forest.grow(rate=8, years=2)
    forest.graph()


if __name__ == '__main__':
    main()

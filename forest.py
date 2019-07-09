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
        self.deku = tree.Tree(location=(0, 0), forest=self, generator=Generator(g=load_model('./saveddekus/DEKU100.h5')), name=self.names.pop(0))
        self.trees = [self.deku]
        self.connections = {self.deku: []}
        self.GANs = {}
        self.diversity = {}
        print("Forest generated!")

    def grow(self, rate=1, years=1):
        for y in range(years):
            r = rate
            self.age()
            for t in self.trees:
                t.getnewneighbors()
            while randbelow(100) < r * 100:
                self.spawn()
                r -= 1
            # self.communicate(plot=(y==years))
            # self.communicate(time=4, plot=True)
            self.graph(name=y+1)
            print('YEAR ' + str(y+1))

    def communicate(self, time=1, plot=False):
        # change time variable for how well trained the tree's generator will be
        for _ in range(time):
            for tr in self.trees:
                for t in self.connections[tr]:
                    if tr.age < t.age and (tr, t) not in self.GANs:
                        self.GANs[(tr, t)] = GAN(generator=tr.generator, discriminator=t.discriminator, nn=t.nn)
                        # tr.communicate(t, plot=plot)
        for gan in self.GANs:
            n = 1
            while tr.name + str(n) in self.diversity:
                n += 1
            self.diversity[tr.name + str(n)] = self.GANs[gan].train(xtrain=produce_language(gan[1].generator.G), epochs=1, tree=gan[0], plot=plot)

        for tree in self.trees:
            tree.resetDiscriminator()

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

    def graph(self, name='test'):
        visualize(self.trees, str(name))


def main():
    forest = Forest()
    forest.grow(rate=1, years=3)
    # forest.grow(rate=0, years=10)
    # forest.grow(rate=0, years=500)
    # forest.graph()
    #forest.grow(rate=8, years=2)
    # with open('languagediversity.txt', 'w') as f:
    #     [f.write(str(i) + ': ' + str(forest.diversity[i]) + '\n') for i in forest.diversity]

if __name__ == '__main__':
    main()

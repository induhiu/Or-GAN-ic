""" A 'forest' class. Represents a collection of 'trees' designed to communicate
    with one another due to their each having the necessary properties of
    Generative Adversarial Nets (GANs).
    Implementation by Kenny Talarico as part of research with Ian Nduhiu
    and advisor David Perkins, June 2019. """

from gan import Generator
from gan import Discriminator
from gan import GAN
from tree import Tree
from random import choice
from secrets import randbelow
from language_getter import produce_language
from graphing import graph as visualize
from tensorflow.keras.models import load_model
from pickle import load

random_dim = 100

class Forest:
    def __init__(self):
        """ Constructor for Forest. """
        print('\n', 'In the vast, deep forest of Hyrule...', '\n',
              'Long have I served as the guardian spirit...', '\n',
              'I am known as the Deku Tree...', '\n\n', 'Creating Deku Tree...',
              '\n', sep='')

        # If names like 'Treebeard' and 'The Giving Tree' are desired, set fun_names
        # to True. Change to False for trees just to have numbers for names.
        fun_names = True
        self.names = (load(open('names.txt', 'rb')) if fun_names else
                      [str(i) for i in range(500)])

        # Create one tree, the Great Deku Tree. He has a saved generator located
        # in the  saveddekus  folder. Here we are loading a generator that has
        # trained for 100 epochs.
        self.trees = [Tree(location=(0, 0), forest=self,
                         generator=Generator(g=load_model('./saveddekus/DEKU100.h5')),
                         name=self.names.pop(0))]
        self.connections = {self.trees[0]: []}
        self.GANs = {}
        print("Forest generated!")

    def grow(self, rate=1, years=1):
        """ The forest 'grows' for a set amount of iterations given in  years  .
            rate=1: exactly one tree is born
            rate=2: exactly two trees are born
            rate=1.4: one tree is born and 40% chance for another birth to occur """

        for y in range(1, years+1):
            r = rate

            # all trees age one year
            self.age()

            # aging means root growth, check for new connections
            for t in self.trees:
                t.getnewneighbors()

            # birthing process using rate
            while randbelow(100) < r * 100:
                self.spawn()
                r -= 1
                
            self.communicate(plot=True)

            # output a graph with the name of the year as identification
            self.graph(name=y)
            print('YEAR ' + str(y))

    def communicate(self, time=1, plot=False):
        """ All trees with connections communicate with one another. """

        # Change time variable for how well trained the tree's generator will be
        for _ in range(time):
            for tr in self.trees:
                for t in self.connections[tr]:
                    if tr.age < t.age and (tr, t) not in self.GANs:
                        self.GANs[(tr, t)] = GAN(generator=tr.generator, discriminator=t.discriminator, nn=t.nn)

        for gan in self.GANs:
            self.GANs[gan].train(xtrain=produce_language(gan[1].generator.G), epochs=1, tree=gan[0], plot=plot)

        # Necessary to avoid overfitting
        for tree in self.trees:
            tree.resetDiscriminator()

    def spawn(self):
        """ This method will take a random tree and induce it to birth a sapling. """

        baby = None

        # This deals with anomalous cases where random choice doesn't work well
        # and a tree has difficulty finding a legal position for a sapling.
        while not baby:
            parent = choice(self.trees)
            while parent.age == 1:
                parent = choice(self.trees)
            baby = parent.spawnChild()
        self.trees.append(baby)

    def age(self):
        """ Age every tree in the forest by one year. """
        for t in self.trees:
            t.age += 1

    def graph(self, name='test'):
        """ Output a graph of the forest. """
        visualize(self.trees, str(name))


def main():
    forest = Forest()
    forest.grow(rate=1, years=3)

if __name__ == '__main__':
    main()

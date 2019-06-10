''' A graph implementation of a forest by Ian Nduhiu, Kenny Talarico and
    Dave Perkins. Created on May 31, 2019. '''

import gan
import tree
from random import choice
import numpy as np

random_dim = 100

class Forest:
    def __init__(self):
        self.dekutree = tree.Tree()
        self.trees = [[None, None, None, None, None],
                      [None, None, None, None, None],
                      [None, None, self.dekutree, None, None],
                      [None, None, None, None, None],
                      [None, None, None, None, None]]

    def addTree(self, tree, r, c):
        self.trees[r][c] = tree


def main():
    forest = Forest()
    forest.addTree(tree.Tree(), 2, 1)
    forest.addTree(tree.Tree(), 2, 3)
    print(forest.trees)



if __name__ == '__main__':
    main()

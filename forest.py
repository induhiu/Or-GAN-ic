''' A graph implementation of a forest by Ian Nduhiu, Kenny Talarico and
    Dave Perkins. Created on May 31, 2019. '''

import gan
import tree
from random import choice
import numpy as np

random_dim = 100

class Forest:
    def __init__(self, size):
        assert size >= 3
        self.dekutree = tree.Tree()
        self.trees = [[None for _ in range(size)] for _ in range(size)]
        self.trees[size // 2][size // 2] = self.dekutree

    def addTree(self, tree, r, c):
        self.trees[r][c] = tree

    def treeHere(self, r, c):
        return self.trees[r][c] is not None


def main():
    forest = Forest(5)
    # forest.addTree(tree.Tree(), 2, 1)
    # forest.addTree(tree.Tree(), 2, 3)
    print(forest.trees)




if __name__ == '__main__':
    main()

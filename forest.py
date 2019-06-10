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
        self._dekutree = tree.Tree(id=(size // 2, size // 2))
        self.trees = [self._dekutree]
        self.forest = [[None for _ in range(size)] for _ in range(size)]
        self.forest[size // 2][size // 2] = self._dekutree
        self.size = size

    def addTreeHere(self, rc):
        assert self.hasNeighbor((rc[0], rc[1]))
        self.forest[rc[0]][rc[1]] = t = tree.Tree(id=rc)
        self.trees.append(t)

    def addTree(self):
        self.addTreeHere(choice([(r, c) for c in range(self.size) for r in range(self.size) if self.hasNeighbor((r, c)) and not self.treeHere((r, c))]))

    def treeHere(self, rc):
        return self.forest[rc[0]][rc[1]] is not None

    def hasNeighbor(self, rc):
        assert 0 <= rc[0] < self.size
        assert 0 <= rc[1] < self.size
        for d in ((1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)):
            if 0 <= rc[0] + d[0] < self.size and 0 <= rc[1] + d[1] < self.size and \
              self.treeHere((rc[0] + d[0], rc[1] + d[1])):
                return True
        return False

    def age(self):
        for t in self.trees:
            t.age += 1

    def __str__(self):
        output = ''
        for li in self.forest:
            for t in li:
                if t is None:
                    output += ('X' + ' ' * 9)
                else:
                    output += (t.__str__() + (10-len(t.__str__())) * ' ')
            output += '\n\n'
        return output


def main():
    forest = Forest(5)
    for _ in range(24):
        forest.age()
        forest.addTree()
    print(forest)

if __name__ == '__main__':
    main()

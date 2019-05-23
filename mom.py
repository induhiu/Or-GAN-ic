"""
Early implementation of a "mom" inventing a "language." The "mom" will do so
teach the language to a "baby." This "mom" uses NumPy arrays of zeros and ones
to represent various English words.
Implementation by Kenny Talarico, 5/23/2019.
"""
import numpy as np
from random import choice as pick

class Logogram:

    def __init__(self, array):
        self.L = array

class Mom:

    def __init__(self, alphabet, signs):
        self.alphabet = alphabet
        self.signs = signs
        # change to alter dimensions of 3x3 array. dim[0] and dim[1] should be
        # the same.
        self.dim = (3, 3)
        self.size = self.dim[0] * self.dim[1]
        self._createLanguage()

    def _createLanguage(self):

        # l is a list of empty Logograms with NumPy arrays of the above dimensions
        logo = [Logogram(np.empty((3, 3), dtype=int)) for _ in range(2 ** self.size)]

        # create every permutation of 0s and 1s
        for x in range(2 ** self.size):
            # change to reflect whatever characters are in alphabet
            bits = createBits(self.size, x)
            for ch in range(len(bits)):
                logo[x].L[ch // self.dim[0]][ch % self.dim[0]] = int(bits[ch])

        self.dictionary = {l:pick(self.signs) for l in logo}

    def __str__(self):
        for ary in self.dictionary:
            print(ary.L, self.dictionary[ary])
        return("--MOM--" + '\n' + "My alphabet consists of " + str(len(self.alphabet)) +
               " characters: " + str(self.alphabet) + '. ' + "I have " + str(2**self.size) +
               " words for " + str(self.signs) + '.')


def createBits(x, num):
    result = "";
    # x is length of bitstring
    v = 2 ** (x - 1)
    for _ in range(x):
      if num >= v:
          result += '1'
          num -= v
      else:
          result += '0'
      v /= 2
    return result


def main():
    # Change to fill array with any other values
    mom = Mom([0, 1], ['red', 'green', 'blue'])
    print(mom)





if __name__ == '__main__':
    main()

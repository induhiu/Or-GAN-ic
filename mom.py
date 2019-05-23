"""
Early implementation of a "mom" inventing a "language." The "mom" will then
teach the language to a "baby." This "mom" uses NumPy arrays of two characters
to represent various English words.
Implementation by Kenny Talarico, 5/23/2019.
"""
import numpy as np
import secrets

class Logogram:

    # Logogram is a class so that the values can be stored in a dictionary
    # later. NumPy arrays can't be dict keys.
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
        """ Create the language. """
        self.dictionary = {}
        used = []

        # create word for each sign
        for s in self.signs:

            # avoid overlap
            n = secrets.randbelow(2 ** self.size)
            while n in used: n = secrets.randbelow(2 ** self.size)
            used.append(n)

            # start with an "empty" logogram
            logo = Logogram(np.empty(self.dim, dtype=str))

            # create a bitstring with the given alphabet
            bits = createBits(self.size, n, self.alphabet)
            for ch in range(len(bits)):
                logo.L[ch // self.dim[0]][ch % self.dim[0]] = bits[ch]

            # update dictionary
            self.dictionary[logo] = s

    def __str__(self):
        """ Provide information about this mom. """
        for ary in self.dictionary:
            print(ary.L, self.dictionary[ary])
        return("My alphabet consists of " + str(len(self.alphabet)) + " characters: "
               + str(self.alphabet) + '. ' + "I have " + str(len(self.signs)) +
               " signs: " + str(self.signs) + '.')


def createBits(size, num, al):
    """ num is the numeral, al is the given alphabet. Will return a bitstring
    in the base equivalent to the size of the alphabet. """
    base = len(al)
    result = '';
    bitstr = ['0' for _ in range(size)]

    # for base 2
    if num % 1 == 0:
        bitstr[7] = '1'
    if num % 2 == 0:
        bitstr[6] = '1'
    if num % 4 == 0:
        bitstr[5] = '1'
    if num % 8 == 0:
        bitstr[4] = '1'
    if num % 16 == 0:
        bitstr[3] = '1'
    if num % 32 == 0:
        bitstr[2] = '1'
    if num % 64 == 0:
        bitstr[1] = '1'
    if num % 128 == 0:
        bitstr[0] = '1'

    print(bitstr)

    return result


def main():
    # Change to fill array with any other values.
    # Alphabet supports two letters. There can be up to 512 signs.
    # mom = Mom(['0', '1'], ['kenny', 'ian', 'dave', 'decker'])
    # print(mom)
    createBits(8, 4, [0, 1])

if __name__ == '__main__':
    main()

"""
Early implementation of a "mom" inventing a "language." The "mom" will then
teach the language to a "baby." This "mom" uses NumPy arrays of characters
to represent various English words.
Implementation by Kenny Talarico, May 2019.
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
            n = secrets.randbelow(len(self.alphabet) ** self.size)
            while n in used: n = secrets.randbelow(len(self.alphabet) ** self.size)
            used.append(n)

            # start with an "empty" logogram
            logo = Logogram(np.empty(self.dim, dtype=str))

            # create a bitstring with the given alphabet
            bits = convertToBase(n, len(self.alphabet), self.alphabet)
            # leading "zeros"
            while len(bits) < self.size:
                bits = self.alphabet[0] + bits

            # convert to array
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


def convertToBase(n, base, alphabet):
    """ Adapted from https://interactivepython.org/runestone/static/pythonds/Recursion/
        pythondsConvertinganIntegertoaStringinAnyBase.html """
    if n < base:
        return alphabet[n]
    else:
        return convertToBase(n // base, base, alphabet) + alphabet[n % base]

def main():
    # Change to fill array with any other values.
    # There can an arbitrarily large alphabet and up to 512 signs.
    mom = Mom(['a', 'b', 'c', 'd', 'e', 'f'], ['kenny', 'ian', 'dave', 'decker'])
    print(mom)

if __name__ == '__main__':
    main()

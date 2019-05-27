"""
Early implementation of a "mom" inventing a "language." The "mom" will then
teach the language to a "baby." This "mom" uses NumPy arrays of characters
to represent various English words.
Implementation by Kenny Talarico, May 2019.
"""
import numpy as np
import secrets
from random import choice


class Mom:

    def __init__(self, alphabet, signs):
        self.alphabet = alphabet
        self.signs = signs
        # self.ASCIIsigns = []
        # ASCIIstr = ''
        # for word in signs:
        #     for letter in word:
        #         temp = str(ord(letter))
        #         while len(temp) < 3:
        #             temp = '0' + temp
        #         ASCIIstr += temp
        #     self.ASCIIsigns.append(ASCIIstr)
        #     ASCIIstr = ''

        # each word will be 9 characters
        self.size = 9
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

            # create a bitstring with the given alphabet
            word = convertToBase(n, len(self.alphabet), self.alphabet)
            # leading "zeros"
            while len(word) < self.size:
                word = self.alphabet[0] + word

            # update dictionary
            self.dictionary[word] = s

    def __str__(self):
        """ Provide information about this mom. """
        for ary in self.dictionary:
            print(ary, self.dictionary[ary])
        return("My alphabet consists of " + str(len(self.alphabet)) + " characters: "
               + str(self.alphabet) + '. ' + "I have " + str(len(self.signs)) +
               " signs: " + str(self.signs) + '.')

    def guess(self, mystery):
        lowest = list(self.dictionary.keys())[0]
        for word in self.dictionary:
            if (hammingDistance(self.dictionary[lowest], mystery) <
                hammingDistance(self.dictionary[word], mystery)):
               lowest = word
        return word

    def speak(self):
        return choice(list(self.dictionary.keys()))


def hammingDistance(str1, str2):
    ham = len(str1)
    for ch in range(len(str1)):
         if str1[ch] == str2[ch]: ham -= 1
    return ham

def convertToBase(n, base, alphabet):
    """ Adapted from https://interactivepython.org/runestone/static/pythonds/Recursion/
        pythondsConvertinganIntegertoaStringinAnyBase.html """
    return (alphabet[n] if n < base else convertToBase(n // base, base, alphabet) + alphabet[n % base])

# def main():
#     # Change to fill array with any other values.
#     # There can an arbitrarily large alphabet and up to 512 signs.
#     mom = Mom(['a', 'b', 'c', 'd', 'e', 'f'], ['kenny', 'ian', 'dave', 'decker'])
#     print(mom)

# if __name__ == '__main__':
#     main()

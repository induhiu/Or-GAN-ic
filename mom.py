"""
Early implementation of a "mom" inventing a "language." The "mom" will do so
teach the language to a "baby." This "mom" uses NumPy arrays of zeros and ones
to represent various English words.
Implementation by Kenny Talarico, 5/23/2019.
"""
import numpy as np

def createBits(x):
    result = "";
    # x is length of bitstring
    v = 2 ** x
    for _ in range(x):
      if x >= v:
        result += '1'
        x -= v
      else:
          result += '0'
      v /= 2
    return result

def createLogograms(A):
    # change to alter dimensions of 3x3 array
    dim = (3, 3)

    size = dim[0] * dim[1]

    # l is a list of empty NumPy arrays of the above dimensions
    l = [np.empty((3, 3), dtype=int) for _ in range 2 ** size]

    for x in range(2 ** (dim[0] * dim[1])):
        for ch in createBits(size):
            for a in range(dim[0]):
                for b in range(dim[1]):
                    l[x][a][b] = ch

    return l

def main():
    # Change to fill array with any other values
    alphabet = [0, 1]

    signs = ['red', 'green', 'blue']

    logograms = createLogograms(alphabet)

if __name__ == '__main__':
    main()

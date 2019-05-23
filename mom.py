"""
Early implementation of a "mom" inventing a "language." The "mom" will do so
teach the language to a "baby." This "mom" uses NumPy arrays of zeros and ones
to represent various English words.
Implementation by Kenny Talarico, 5/23/2019.
"""
import numpy as np

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

def createLogograms(A):
    # change to alter dimensions of 3x3 array. dim[0] and dim[1] should be
    # the same.
    dim = (3, 3)

    size = dim[0] * dim[1]

    # l is a list of empty NumPy arrays of the above dimensions
    l = [np.empty((3, 3), dtype=int) for _ in range(2 ** size)]

    # create every permutation of 0s and 1s
    for x in range(2 ** (dim[0] * dim[1])):
        bits = createBits(size, x)
        for ch in range(len(bits)):
            l[x][ch // dim[0]][ch % dim[0]] = int(bits[ch])
    return l

def main():
    # Change to fill array with any other values
    alphabet = [0, 1]

    signs = ['red', 'green', 'blue']

    # logograms = createLogograms(alphabet)

    # print(logograms[511])

    for x in range(256):
        print(createBits(8, x))

if __name__ == '__main__':
    main()

""" Script to convert a large dataset of images into a text file consisting
    of representations of NumPy arrays.
    Implementation by Kenny Talarico, June 2019. """

from numpy import array
from PIL import Image

def writetofile(ary, out):
    out.write('\n'.join(' '.join(str(255-c) for c in r) for r in ary))
    out.write('\n')

def main():
    ary = array(Image.open('testin.png'))
    out = open('testout.txt', 'w')

    for _ in range(70000):
        writetofile(ary, out)

    #Image.fromarray(ary).save('testout.png')

if __name__ == '__main__':
    main()

""" Script to convert a large dataset of images into an npz file consisting
    of representations of NumPy arrays.
    Implementation by Kenny Talarico, June 2019. """

import numpy as np
from Pillow import Image
import os

def main():
    dir = os.getcwd() + '/logograms'
    imgs = os.listdir(dir)
    np.savez_compressed('imgarys.npz', **{imgs[i][:5]: np.array(Image.open(dir + '/' + imgs[i])) for i in range(70000)})

if __name__ == '__main__':
    main()

''' Figuring out how text preprocessing works '''

import os
import tensorflow
import keras
import numpy as np

# Let Keras know that we are using tensorflow as our backend engine
os.environ["KERAS_BACKEND"] = "tensorflow"

# To make sure that we can reproduce the experiment and get the same results
np.random.seed(10)

def main():
    ''' Main Function '''
    string = "My name is Ian Nderi Nduhiu"
    n = 200  # Size of number vocabulary [1, n]

    # Using one hot encoding, which returns a list of integers that
    # represent our string, but encoded.
    encoded = keras.preprocessing.text.one_hot(string, n,
                                    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                    lower=False, split=' ')
    print(encoded)

if __name__ == '__main__':
    main()

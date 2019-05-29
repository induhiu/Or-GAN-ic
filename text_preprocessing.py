''' Figuring out how text preprocessing works '''

import os
import tensorflow
import keras
import numpy as np
from math import factorial
import random

# Let Keras know that we are using tensorflow as our backend engine
os.environ["KERAS_BACKEND"] = "tensorflow"

# To make sure that we can reproduce the experiment and get the same results
np.random.seed(10)

def get_words():
    ''' Gets common(ish) English words from a text file '''
    string = ""
    with open('english_words.txt') as english_words:
        list_of_words = english_words.readlines()
    list_of_words = [word[: -1] for word in list_of_words]
    for i in range(random.randint(1, 50)):
        string += random.choice(list_of_words)
        string = string + " " if i < 100 else string + ''
    return string

def main():
    ''' Main Function '''
    string = get_words()
    n = factorial(9) # Size of number vocabulary [1, n]
    print(string)

    # Using one hot encoding, which returns a list of integers that
    # represent our string, but encoded.
    encoded = keras.preprocessing.text.one_hot(string, n, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=False, split=' ')
    print(encoded)

    # # Using text to word sequence, let's see how this goes
    # string = "My name is Moana"
    # encoding = keras.preprocessing.text.text_to_word_sequence(string, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=False, split=' ')
    # print("this is text_to_word_sequence: \n", encoding)
    # # encoding becomes a list of the individual words in the string

    # # Using hashing trick, let's see how this goes
    # string = "My name is Maui"
    # encoding = keras.preprocessing.text.hashing_trick(string, n, hash_function=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=False, split=' ')
    # print("this is hashing encoding: \n", encoding)
    # # More or less the same output as one-hot encoding but uses hashing





if __name__ == '__main__':
    main()

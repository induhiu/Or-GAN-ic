alphabets = 'abcdefghijklmnopqrstuvwxyz'

from math import factorial
import random
import language_gan
import keras
import os
import tensorflow

os.environ['KERAS_BACKEND'] = "tensorflow"

def get_words():
    ''' Returns one-hot encoded words and their counterparts '''
    with open('english_words.txt') as english_words:
        list_of_words = english_words.readlines()
    list_of_words = [word[: -1] for word in list_of_words]
    chosen_words = [random.choice(list_of_words) for _ in range(10000)]
    encoded = [keras.preprocessing.text.one_hot(x, factorial(9),
               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=False,
               split=' ') for x in chosen_words]
    return (chosen_words, encoded)




def main():
    ''' Main function '''
    words, one_hot_encoding = get_words()
    print(words)
    print(one_hot_encoding)

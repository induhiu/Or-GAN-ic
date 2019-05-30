from math import factorial
from collections import Counter
import itertools
import random
import language_gan
import keras
import os
import tensorflow
import numpy as np
import sys

os.environ['KERAS_BACKEND'] = "tensorflow"

def get_words():
    ''' Returns one-hot encoded words and their counterparts '''
    with open('english_words.txt') as english_words:
        list_of_words = english_words.readlines()
    list_of_words = [word[: -1] for word in list_of_words]
    random_words = [random.choice(list_of_words) for _ in range(78400)]
    encoded = [keras.preprocessing.text.one_hot(x, factorial(9),
               filters='\t\n', lower=False,
               split=' ') for x in random_words]
    return (random_words, encoded)

def score(original, prediction):
    ''' Scores our prediction by checking for values within 20 % '''
    in_original = 0
    for val in range(len(prediction)):
        percentages = np.array([abs(prediction[val] / x) for x in original\
                                if 0.995 <= abs(prediction[val] / x) <= 1.005])
        print(percentages)
        input()
        if len(percentages) > 1:
            in_original += 1
        # for p in percentages:
        #     if 0.9995 <= p <= 1.0005:
        #         in_original += 1
        #         break
    return in_original

def main():
    ''' Main function '''
    words, one_hot_encoding = get_words()
    ones = [-1, 1]
    one_hot_encoding = list(itertools.chain.from_iterable(one_hot_encoding))
    lst_one_hot_encoding = [((x / factorial(9)) * random.choice(ones)) for x in one_hot_encoding]
    np_one_hot_encoding = np.array(lst_one_hot_encoding).reshape(100, 784)
    my_gan = language_gan.GAN(np_one_hot_encoding)
    my_gan.train()
    prediction = my_gan.predict()
    # # print(prediction)
    lst_prediction = list(prediction)
    lst_prediction = list(itertools.chain.from_iterable([list(x) for x in lst_prediction]))
    sample = [random.choice(lst_prediction) for _ in range(10)]
    gan_score = score(lst_one_hot_encoding, sample)
    print(gan_score)


if __name__ == '__main__':
    main()

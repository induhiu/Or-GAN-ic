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
    ''' Returns one-hot encoded words and their counterpart words '''
    with open('english_words.txt') as english_words:
        list_of_words = english_words.readlines()
    # Removes the newline character at the end of each word
    list_of_words = [word[: -1] for word in list_of_words]
    # Choose 78400 words
    random_words = [random.choice(list_of_words) for _ in range(78400)]
    encoded = [keras.preprocessing.text.one_hot(x, factorial(9),
               filters='\t\n', lower=False,
               split=' ') for x in random_words]
    return (random_words, encoded)

def get_score(original, prediction):
    ''' Scores our prediction by checking for values within n %. n is arbitrarily
       set. '''
    n = 0.10
    in_original = 0
    for val in range(len(prediction)):
        percentages = np.array([abs(prediction[val] / x) for x in original\
                                if 1 - n <= abs(prediction[val] / x) <= 1 + n])
        if len(percentages) > 0: # if at least one word was correctly predicted
            in_original += 1
    return in_original

def score(original, prediction):
    ''' Scores the gan for human evaluation '''
    # Pick random words from sample space(prediction)
    prev = [random.choice(prediction) for _ in range(10)]
    gan_score = get_score(original, prev)
    print(gan_score)
    for _ in range(10):  # ten sampling times
        samples = []
        for _ in range(10):
            while True:
                x = random.choice(prediction)
                if x not in prev:  # ensures we do not repeat a value
                    samples.append(x)
                    break
        print(get_score(original, samples))
        prev.extend(samples)  # add used values to container list
        input('Continue? ')

def main():
    ''' Main function '''
    words, one_hot_encoding = get_words()
    ones = [-1, 1]  # to randomly set negative and non-negative numbers

    # Convert the one_hot_encoding to a flattened list
    one_hot_encoding = list(itertools.chain.from_iterable(one_hot_encoding))

    # Convert the numbers to ensure they are in range (-1, 1)
    lst_one_hot_encoding = [((x / factorial(9)) * random.choice(ones)) for x in one_hot_encoding]

    # Convert to np array and reshape
    np_one_hot_encoding = np.array(lst_one_hot_encoding).reshape(100, 784)

    # Initialize the gan
    my_gan = language_gan.GAN(np_one_hot_encoding)
    my_gan.train()
    prediction = my_gan.predict()  # get the gan to produce words
    lst_prediction = list(prediction)

    # To flatten the list into one big ol' list
    lst_prediction = list(itertools.chain.from_iterable([list(x) for x in lst_prediction]))

    # If you want to score the model, uncomment the line below
    # score(lst_one_hot_encoding, lst_prediction)


if __name__ == '__main__':
    main()

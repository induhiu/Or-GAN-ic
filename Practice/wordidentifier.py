""" Implementation for a neural network that assigns meaning to words given
    a training set considering what the words "really" mean. It updates
    itself with each generation in order to maintain the flexibility of the
    language.
    Implementation by Kenny Talarico, May 2019. """

import tensorflow
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import preprocessing

class WordIdentifier:

    def __init__(self, lin):
        self.dict = {line[line.rfind(' ')+1:]: line[:line.rfind(' ')].split() for line in lin}

    def __str__(self):
        return self.dict

    def output(self):
        longest = 1
        for k in self.dict:
            if len(k) > longest:
                longest = len(k)
        for k in self.dict:
            print(k + ': ', end=' '*((longest+1)-len(k)))
            for v in self.dict[k]:
                print(v, end=' | ')
            print()

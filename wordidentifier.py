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

    def __init__(self):
        self.dict = {line[line.rfind(' ')+1:]: line[:line.rfind(' ')].split() for line in
                     [w[:-1] for w in open('momwords.txt', 'r').readlines()]}

        # keys = [preprocessing.text.one_hot(k, 2000000)[0] for k in list(dict.keys())]
        # vals = [preprocessing.text.one_hot(v, 2000000)[0] for v in list(dict.values())]
        #
        # newDict = {keys[i]: vals[i] for i in range(len(keys))}
        #
        # model = Sequential()
        # model.add(Dense(10, activation='relu', input_dim=1))
        # model.add(Dense(6 + len(dict), activation='relu'))
        # model.add(Dense(len(dict), activation='softmax'))
        #
        # xtrain, ytrain = np.array(list(newDict.keys())), np.array(list(newDict.values()))
        #
        # ytrain = np.arange(0, len(ytrain)).reshape(len(keys), 1)
        #
        # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
        #
        # model.fit(xtrain, ytrain, epochs=10, batch_size=5)

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


if __name__ == '__main__':
    main()

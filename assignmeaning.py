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

def main():
    fi = [w[:-1] for w in open('momwords.txt', 'r').readlines()]
    dict = {word[:word.index(':')]: word[word.index(':') + 1:] for word in fi}

    keys = [preprocessing.text.one_hot(k, 2000000)[0] for k in list(dict.keys())]
    vals = [preprocessing.text.one_hot(v, 2000000)[0] for v in list(dict.values())]

    newDict = {keys[i]: vals[i] for i in range(len(keys))}

    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=1))
    model.add(Dense(6 + len(dict), activation='relu'))
    model.add(Dense(len(dict), activation='softmax'))

    xtrain, ytrain = np.array(list(newDict.keys())), np.array(list(newDict.values()))

    # ytrain = np.arange(0, len(ytrain)).reshape(len(keys), 1)

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])

    model.fit(xtrain, ytrain, epochs=100, batch_size=5)


if __name__ == '__main__':
    main()

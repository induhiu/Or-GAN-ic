''' Neural network implementation. Will be used to give meaning to images.
    Documentation(some) by Ian Nduhiu. Find an example implementation at the
    bottom of the code'''
# Adapted from https://nextjournal.com/gkoehler/digit-recognition-with-keras

# imports for array-handling and plotting
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
# for testing on CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

# keras imports for the dataset and building our neural network
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

# for reading data from text file
import pickle

import gan
from language_getter import produce_language

from collections import Counter
# for exits while debugging, use sys.exit()
import sys

alphabets = 'ABCEDFGHIJ'

def load_external_data():
    ''' Returns externally loaded datasets for training neural network '''
    all_vals = pickle.load(open('lang_for_nn.txt', 'rb'))
    xtrain, xtest = np.array([x[0] for x in all_vals][:60000]),\
                        np.array([x[0] for x in all_vals][60000:])
    ytrain, ytest = np.array([alphabets.index(x[1]) for x in all_vals][:60000]),\
                        np.array([alphabets.index(x[1]) for x in all_vals][60000:])
    xtrain = xtrain.reshape(60000, 784)
    xtest = xtest.reshape(10000, 784)
    return (xtrain, ytrain, xtest, ytest)


class Neural():
    def __init__(self):
        ''' The Constructor '''
        self.x_train, self.y_train, self.x_test, self.y_test = load_external_data()
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255

        # Encoding labels. Example 4 becomes [0,0,0,0,1,0,0,0,0,0]
        n_classes = 10
        self.y_train = np_utils.to_categorical(self.y_train, n_classes)
        self.y_test = np_utils.to_categorical(self.y_test, n_classes)


        # building a linear stack of layers with the sequential model
        self.model = Sequential()
        self.model.add(Dense(512, input_shape=(784,)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))

        # compiling the sequential model
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                        optimizer='adam')

    def train_model(self, e=1):
        ''' Trains the model '''
        # Default epoch is set to 1
        self.model.fit(self.x_train, self.y_train, epochs=e, batch_size=128)

    def give_meaning(self, data=None):
        ''' Predicts meaning of symbols. Returns an array of predictions '''
        return self.model.predict(self.x_test) if data is None\
            else self.model.predict(data)

    def get_count(self, language):
        ''' Returns individual counts of the testing data'''
        pred = self.give_meaning(language)
        # Create a counter object to count each entry
        my_counter = Counter([alphabets[list(x).index(max(x))] for x in pred])
        return my_counter

#  ------------------------------------------------------------------------  #
# # This is an example of how to use the neural network in our gan
# # implementation. You can use it for your reference
# if __name__ == '__main__':
    # # Creating a gan
    # my_gan = gan.GAN()
    # my_gan.train(epochs=10, plot=False)
    # # Creating a neural network object
    # nn = Neural()
    # nn.train_model(e=1)
    # # Get how many of each are produced by a gan
    # my_counter = nn.get_count(produce_language(my_gan.G).reshape(60000, 784))
    # print(my_counter)

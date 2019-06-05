''' Neural network implementation. Will be used to give meaning to images.
    Documentation(some) by Ian Nduhiu'''
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

class Neural():
    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None):
        ''' The Constructor '''
        if x_train is None:  # if data is not loaded externally
            self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()
        else:
            self.x_train = x_train
            self.y_train = y_train
            self.x_test = x_test
            self.y_test = y_test

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

    def load_data(self):
        ''' Loads data from Mnist if data is not provided '''
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # Reshape x_train and y_train into 2D arrays
        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        # normalizing the data to help with the training
        X_train /= 255
        X_test /= 255

        # one-hot encoding using keras' numpy-related utilities
        n_classes = 10
        Y_train = np_utils.to_categorical(y_train, n_classes)
        Y_test = np_utils.to_categorical(y_test, n_classes)

        return (X_train, Y_train, X_test, Y_test)

    def train_model(self, e=1):
        ''' Trains the model '''
        self.model.fit(self.x_train, self.y_train, epochs=e, batch_size=128)

    def give_meaning(self):
        ''' Predicts meaning of symbols '''
        return self.model.predict(self.x_test)

if __name__ == '__main__':
    nn = Neural()
    nn.train_model()
    pred = nn.give_meaning()
    # graph(nn.x_test)
    for i in range(10):
        print(pred[i], nn.y_test[i])

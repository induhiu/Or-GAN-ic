# Taken from https://www.datacamp.com/community/tutorials/generative-adversarial-networks

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.keras.layers.core import Dense, Dropout
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
from math import factorial
import sys

# Let Keras know that we are using tensorflow as our backend engine
os.environ["KERAS_BACKEND"] = "tensorflow"

# To make sure that we can reproduce the experiment and get the same results
np.random.seed(10)

# The dimension of our random noise vector.
random_dim = 100

class GAN:
    def __init__(self, data=None):
        self.data = data
        self.adam = self.get_optimizer()
        # This is bad styling. Fix later
        self.generator = self.get_generator(self.adam)
        self.discriminator = self.get_discriminator(self.adam)
        self.gan = self.get_gan_network(self.discriminator, random_dim,
                                        self.generator, self.adam)

    def load_data(self):
        ''' Gets the language to be used as training data. Can receive an array
        of words from the language generated from the two moms. For now, it
        will use randomly created arrays for testing '''
        n = factorial(9)
        x_train = self.data # data passed from outside function

        # The ones below do not really matter as this is unsupervised learning
        y_train = np.random.randint(1, n, size=(10000))
        x_test = np.random.randint(1, n, size=(100, 784))
        y_test = np.random.randint(1, n, size=(100))

        return (x_train, y_train, x_test, y_test)

    def predict(self, n=factorial(9)):
        ''' Returns a prediction that will be used to score the model '''
        return self.generator.predict(np.random.normal(0, 1, size=[random_dim, random_dim]))

    def get_optimizer(self):
        return Adam(lr=0.0002, beta_1=0.5)

    def get_generator(self, optimizer):
        generator = Sequential()
        generator.add(Dense(256, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        generator.add(LeakyReLU(0.2))

        generator.add(Dense(512))
        generator.add(LeakyReLU(0.2))

        generator.add(Dense(1024))
        generator.add(LeakyReLU(0.2))

        generator.add(Dense(784, activation='tanh'))
        generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        return generator

    def get_discriminator(self, optimizer):
        discriminator = Sequential()
        discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Dense(512))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Dense(256))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Dense(1, activation='sigmoid'))
        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
        return discriminator

    def get_gan_network(self, discriminator, random_dim, generator, optimizer):
        # We initially set trainable to False since we only want to train either the
        # generator or discriminator at a time
        discriminator.trainable = False
        # gan input (noise) will be 100-dimensional vectors
        gan_input = Input(shape=(random_dim,))
        # the output of the generator (an image)
        x = generator(gan_input)
        # get the output of the discriminator (probability if the image is real or not)
        gan_output = discriminator(x)
        gan = Model(inputs=gan_input, outputs=gan_output)
        gan.compile(loss='binary_crossentropy', optimizer=optimizer)
        return gan

    def train(self, epochs=5, batch_size=10):
        # Get the training and testing data
        x_train, y_train, x_test, y_test = self.load_data()
        # Split the training data into batches of size batch_size
        batch_count = x_train.shape[0] // batch_size
        # n = factorial(9

        for e in range(1, epochs+1):
            print('-'*15, 'Epoch %d' % e, '-'*15)
            for _ in tqdm(range(batch_count)):
                # Draws random samples from a normal(Gaussian) distribution
                noise = np.random.normal(0, 1, size=[batch_size, random_dim])
                language_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

                # Generate fake words(noise)
                generated_words = self.generator.predict(noise)
                X = np.concatenate([language_batch, generated_words])

                # Labels for generated and real data
                y_dis = np.zeros(2*batch_size)
                # One-sided label smoothing
                y_dis[:batch_size] = 0.9

                # Train discriminator
                self.discriminator.trainable = True
                self.discriminator.train_on_batch(X, y_dis)

                # Train generator
                noise = np.random.normal(0, 1, size=[batch_size, random_dim])
                y_gen = np.ones(batch_size)
                self.discriminator.trainable = False
                self.gan.train_on_batch(noise, y_gen)

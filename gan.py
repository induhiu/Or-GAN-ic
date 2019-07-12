""" Implementation of a Generative Adversarial Network. Code adapted from
    datacamp tutorials with some additions to cater for our forest
    implementation. """
# Adapted from https://www.datacamp.com/community/tutorials/generative-adversarial-networks

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import choice

import tensorflow

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.keras.layers.core import Dense, Dropout
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers

from tensorflow.keras.models import load_model
from language_getter import produce_language
from pickle import load
from random import shuffle
from collections import Counter

# Let Keras know that we are using tensorflow as our backend engine
os.environ["KERAS_BACKEND"] = "tensorflow"

# The dimension of our random noise vector.
random_dim = 100

# Load a neural network
my_nn = load_model('new_nn.h5')

class Generator:
    def __init__(self, g=None):
        if g:
            self.G = g  # if a generator is exclusively passed
        else:
            self.G = Sequential()
            self.G.add(Dense(256, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
            self.G.add(LeakyReLU(0.2))

            self.G.add(Dense(512))
            self.G.add(LeakyReLU(0.2))

            self.G.add(Dense(1024))
            self.G.add(LeakyReLU(0.2))

            self.G.add(Dense(784, activation='tanh'))
            self.G.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

class Discriminator:
    def __init__(self):
        self.D = Sequential()
        self.D.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.3))

        self.D.add(Dense(512))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.3))

        self.D.add(Dense(256))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.3))

        self.D.add(Dense(1, activation='sigmoid'))
        self.D.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

        self.D.trainable = False

    def reset(self):
        self.D.reset_states()  # resets weights and biases of model

class GAN:
    def __init__(self, nn, random_dim=100, discriminator=None, generator=None):
        self.O = Adam(lr=0.0002, beta_1=0.5)  # Adam optimizer
        self.D = (Discriminator().D if not discriminator else discriminator.D)
        self.G = (Generator().G if not generator else generator.G)
        self.nn = nn  # neural network
        self.input = Input(shape=(random_dim,))
        self.output = self.D(self.G(self.input))

        self.GAN = Model(inputs=self.input, outputs=self.output)
        self.GAN.compile(loss='binary_crossentropy', optimizer=self.O,
                         metrics=['accuracy'])

    def train(self, testid, epochs=1, batch_size=128, id=1, plot=True,
            attack=False, tree=None, xtrain=None, xtest=None):

        # Get the training and testing data
        x_train, y_train, x_test, y_test = 0, 0, 0, 0
        if xtrain is not None:  # if loaded externally
            x_train, x_test = xtrain, xtest
            x_train = (x_train.astype(np.float32) - 127.5)/127.5
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] ** 2)
        else:  # get mnist data
            x_train, y_train, x_test, y_test = load_mnist_data()

        # Split the training data into batches of size 128
        if x_train.shape[0] >= 128:
            batch_count = x_train.shape[0] // batch_size
        else:
            batch_count = 1

        # Initializing the variable
        generated_images = None

        # # Empty list to hold old images for testing out experience replay
        old_imgs = []

        # # A list to hold possible morphed images
        morphed = []

        for e in range(1, epochs+1):
            print('-'*15, 'Epoch %d' % id, '-'*15)
            for _ in tqdm(range(batch_count)):
                noise = np.random.normal(0, 1, size=[batch_size, random_dim])
                image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
                image_batch = image_batch.reshape(128, 784)  # remove later

                # Generate fake MNIST images
                generated_images = self.G.predict(noise)
                X = np.concatenate([image_batch, generated_images])
                y_dis = np.zeros(2*batch_size)

                # One-sided label smoothing
                y_dis[:batch_size] = 0.9

                # Train discriminator
                self.toggleDTrain()
                self.D.train_on_batch(X, y_dis)

                # Train generator
                noise = np.random.normal(0, 1, size=[batch_size, random_dim])
                self.toggleDTrain()

                # Train the GAN model
                self.GAN.train_on_batch(noise, np.ones(batch_size))

            # Plots the images if plot is set to True(default)
            # Can add an extra condition e.g. if id == 10
            # possible_morphs = []
            if plot and not tree:  # plotting images for a normal gan
                plot_generated_images(id, self.G)
            if plot and tree and id % 5 == 0:  # plotting images for forest gan
                plot_tree_images(tree, testid)

            # Increase id
            id += 1

    def toggleDTrain(self):
        self.D.trainable = not self.D.trainable

def load_mnist_data():
    # load the data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # reshape the xtrain
    x_train = reshape_x(x_train)
    x_test = x_test.reshape(10000, 784)
    return (x_train, y_train, x_test, y_test)

def reshape_x(x_train):
    # normalize our inputs to be in the range[-1, 1]
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    # convert x_train with a shape of (60000, 28, 28) to (60000, 784) so we have
    # 784 columns per row
    x_train = x_train.reshape(60000, 784)
    return x_train

def plot_generated_images(id, generator, examples=100, dim=(10, 10),
                        figsize=(10, 10)):
    ''' Plots gan images '''
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    newfile = filename = 'gan_morphs/GANGeneratedImage%d' % id
    copy = 1
    while os.path.exists(newfile + '.png'):
        newfile = filename + '(' + str(copy) + ')'
        copy += 1
    plt.savefig(newfile)
    plt.close('all')

def plot_tree_images(tree, id):
    ''' Plots forest(networked GANs) images '''
    noise = np.random.normal(0, 1, size=[100, 100])
    generated_images = tree.generator.G.predict(noise)
    generated_images = generated_images.reshape(100, 28, 28)
    plt.figure(figsize=(10, 10))
    for i in range(generated_images.shape[0]):
        plt.subplot(10, 10, i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    newfile = filename = 'tree_images/test' + str(id) + '/' + tree.name
    num = 1
    while os.path.exists(newfile + str(num) + '.png'):
        num += 1
    plt.savefig(newfile + str(num) + '.png')
    plt.close('all')

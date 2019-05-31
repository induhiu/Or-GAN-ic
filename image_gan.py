# Adapted from https://www.datacamp.com/community/tutorials/generative-adversarial-networks

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

import tensorflow

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.keras.layers.core import Dense, Dropout
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers

# Let Keras know that we are using tensorflow as our backend engine
os.environ["KERAS_BACKEND"] = "tensorflow"

# To make sure that we can reproduce the experiment and get the same results
np.random.seed(10)

# The dimension of our random noise vector.
random_dim = 100

class GAN:
    def __init__(self, random_dim, x=None, y=None):
        self.O = Adam(lr=0.0002, beta_1=0.5)
        self.D = get_discriminator(self.O)
        self.G = get_generator(self.O)

        self.input = Input(shape=(random_dim,))
        self.output = self.D(self.G(self.input))
        self.curr_x_train, self.curr_y_train = x, y

        self.GAN = Model(inputs=self.input, outputs=self.output)
        self.GAN.compile(loss='binary_crossentropy', optimizer=self.O)

    def train(self, epochs=1, batch_size=128, data_loaded=False):
        # Get the training and testing data
        x_train, y_train, x_test, y_test = 0, 0, 0, 0
        if data_loaded:
            x_train, y_train = self.curr_x_train, self.curr_y_train
        else:
            x_train, y_train, x_test, y_test = load_minst_data()
        # \Split the training data into batches of size 128
        if x_train.shape[0] >= 128:
            batch_count = x_train.shape[0] // batch_size
        else:
            batch_count = 1

        for e in range(1, epochs+1):
            print('-'*15, 'Epoch %d' % e, '-'*15)
            for _ in tqdm(range(batch_count)):
                # Get a random set of input noise and images
                noise = np.random.normal(0, 1, size=[batch_size, random_dim])
                image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
                image_batch = image_batch.reshape(128, 784)  # remove later

                # Generate fake MNIST images
                generated_images = self.G.predict(noise)
                X = np.concatenate([image_batch, generated_images])

                # Labels for generated and real data
                y_dis = np.zeros(2*batch_size)
                # One-sided label smoothing
                y_dis[:batch_size] = 0.9

                # Train discriminator
                self.toggleDTrain()
                self.D.train_on_batch(X, y_dis)

                # Train generator
                noise = np.random.normal(0, 1, size=[batch_size, random_dim])
                self.toggleDTrain()

                self.GAN.train_on_batch(noise, np.ones(batch_size))
            plot_generated_images(e, self.G, self)

    def toggleDTrain(self):
        self.D.trainable = not self.D.trainable

def load_minst_data():
    # load the data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalize our inputs to be in the range[-1, 1]
    x_train = reshape_x(x_train)
    return (x_train, y_train, x_test, y_test)

def reshape_x(x_train):
    # normalize our inputs to be in the range[-1, 1]
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    # convert x_train with a shape of (60000, 28, 28) to (60000, 784) so we have
    # 784 columns per row
    x_train = x_train.reshape(60000, 784)
    return x_train


def get_generator(optimizer):
    G = Sequential()
    G.add(Dense(256, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    G.add(LeakyReLU(0.2))

    G.add(Dense(512))
    G.add(LeakyReLU(0.2))

    G.add(Dense(1024))
    G.add(LeakyReLU(0.2))

    G.add(Dense(784, activation='tanh'))
    G.compile(loss='binary_crossentropy', optimizer=optimizer)
    return G

def get_discriminator(optimizer):
    D = Sequential()
    D.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    D.add(LeakyReLU(0.2))
    D.add(Dropout(0.3))

    D.add(Dense(512))
    D.add(LeakyReLU(0.2))
    D.add(Dropout(0.3))

    D.add(Dense(256))
    D.add(LeakyReLU(0.2))
    D.add(Dropout(0.3))

    D.add(Dense(1, activation='sigmoid'))
    D.compile(loss='binary_crossentropy', optimizer=optimizer)

    D.trainable = False

    return D

# Create a wall of generated MNIST images
def plot_generated_images(epoch, generator, GAN=None, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)
    GAN.curr_x_train = generated_images
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)

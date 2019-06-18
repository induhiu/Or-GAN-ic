# Adapted from https://www.datacamp.com/community/tutorials/generative-adversarial-networks

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from PIL import Image
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
import language_getter
import pickle


# Let Keras know that we are using tensorflow as our backend engine
os.environ["KERAS_BACKEND"] = "tensorflow"

# To make sure that we can reproduce the experiment and get the same results
np.random.seed(10)

# The dimension of our random noise vector.
random_dim = 100

class Generator:
    def __init__(self, optimizer=Adam(lr=0.0002, beta_1=0.5), g=None):
        if g:
            self.G = g
        else:
            self.G = Sequential()
            self.G.add(Dense(256, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
            self.G.add(LeakyReLU(0.2))

            self.G.add(Dense(512))
            self.G.add(LeakyReLU(0.2))

            self.G.add(Dense(1024))
            self.G.add(LeakyReLU(0.2))

            self.G.add(Dense(784, activation='tanh'))
            self.G.compile(loss='binary_crossentropy', optimizer=optimizer)

class Discriminator:
    def __init__(self, optimizer=Adam(lr=0.0002, beta_1=0.5)):
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
        self.D.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.D.trainable = False

    def reset(self):
        self.__init__()

class GAN:
    def __init__(self, random_dim=100, x_train=None, x_test=None, discriminator=Discriminator(Adam(lr=0.0002, beta_1=0.5)),
                 generator=Generator(Adam(lr=0.0002, beta_1=0.5))):
        self.O = Adam(lr=0.0002, beta_1=0.5)
        self.D = discriminator.D
        self.G = generator.G

        self.input = Input(shape=(random_dim,))
        self.output = self.D(self.G(self.input))
        self.curr_x_train, self.curr_x_test = x_train, x_test

        self.GAN = Model(inputs=self.input, outputs=self.output)
        self.GAN.compile(loss='binary_crossentropy', optimizer=self.O,
                         metrics=['accuracy'])

    def train(self, epochs=1, batch_size=128, id=1, plot=True,
            attack=False, all_generated_images=[]):
        # Get the training and testing data
        x_train, y_train, x_test, y_test = 0, 0, 0, 0
        if self.curr_x_train is not None:
            x_train, x_test = self.curr_x_train, self.curr_x_test
            x_train = (x_train.astype(np.float32) - 127.5)/127.5
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] ** 2)
        else:
            x_train, y_train, x_test, y_test = load_minst_data()
        # \Split the training data into batches of size 128
        # print(x_train)
        if x_train.shape[0] >= 128:
            batch_count = x_train.shape[0] // batch_size
        else:
            batch_count = 1

        generated_images = None

        # # Testing out experience replay
        # old_imgs = []
        for e in range(1, epochs+1):
            print('-'*15, 'Epoch %d' % id, '-'*15)
            for _ in tqdm(range(batch_count)):
                # Get a random set of input noise and images
                noise = np.random.normal(0, 1, size=[batch_size, random_dim])
                image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
                image_batch = image_batch.reshape(128, 784)  # remove later

                # Generate fake MNIST images
                generated_images = self.G.predict(noise)

                # new_noise = np.random.normal(0, 1, size=[batch_size, 784])

                # X = np.concatenate([image_batch, new_noise, generated_images])
                X = np.concatenate([image_batch, generated_images])
                y_dis = np.zeros(2*batch_size)
                # experience_rep = []
                # if e != 0 and e % 5 == 0:
                #     x = np.array(old_imgs[-4:]).reshape(800, 784)
                #     experience_rep = np.array([choice(x) for _ in range(128)])
                # #
                # #     # print()
                # #     # print(x.shape)
                # #     # print(experience_rep.shape)
                # #     # sys.exit()
                #     X = np.concatenate([image_batch, experience_rep, generated_images])
                #     y_dis = np.zeros(3*batch_size)
                # Labels for generated and real data
                # y_dis = np.zeros(2*batch_size)

                # y_dis = np.zeros(3*batch_size)
                # One-sided label smoothing
                y_dis[:batch_size] = 0.9

                # Train discriminator
                self.toggleDTrain()
                self.D.train_on_batch(X, y_dis)

                # Train generator
                noise = np.random.normal(0, 1, size=[batch_size, random_dim])
                self.toggleDTrain()

                self.GAN.train_on_batch(noise, np.ones(batch_size))

            # If you want to evaluate the model's perfomance. To be used
            # in times of attack. Threshold loss is 20. If average loss is above
            # threshold, it's most likely a virus. Returns a list indicating
            # loss and accuracy
            eval = self.GAN.evaluate(x=x_test, y=y_test, verbose=0) if attack \
                    else None
            if plot:
                all_generated_images.append(plot_generated_images(id, self.G))
            # if e > 0:
            # old_imgs.append(language_getter.produce_language(self.G, n=2).reshape(200, 784))
            id += 1
        return all_generated_images

    def toggleDTrain(self):
        self.D.trainable = not self.D.trainable

def load_minst_data():
    # load the data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # reshape the xtrain
    x_train = reshape_x(x_train)
    x_test = x_test.reshape(10000, 784)
    x_test = np.array([x_test[i][:100] for i in range(len(x_test))])
    return (x_train, y_train, x_test, y_test)

def reshape_x(x_train):
    # normalize our inputs to be in the range[-1, 1]
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    # convert x_train with a shape of (60000, 28, 28) to (60000, 784) so we have
    # 784 columns per row
    x_train = x_train.reshape(60000, 784)
    return x_train

# Create a wall of generated MNIST images
def plot_generated_images(id, generator, examples=100, dim=(10, 10),
                        figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    newfile = filename = 'GANGeneratedImage%d' % id
    copy = 1
    while os.path.exists(newfile + '.png'):
        newfile = filename + '(' + str(copy) + ')'
        copy += 1
    plt.savefig(newfile)
    plt.close('all')
    return generated_images

if __name__ == '__main__':
#     # GAN().train(epochs=20)
    vals = np.array(pickle.load(open('lang_for_gan.txt', 'rb'))[:60000])
    my_gan = GAN(x_train=vals)
    my_gan.train(epochs=30)
    gan2 = GAN(x_train=language_getter.produce_language(my_gan.G))
    gan2.train(epochs=20)
    gan3 = GAN(x_train=language_getter.produce_language(gan2.G))
    gan3.train(epochs=20)
    # with open('counter.txt', 'wb') as file:
        # pickle.dump(gan.train(epochs=10, plot=False), file)
    # gan.train(epochs=10, plot=False)

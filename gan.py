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
from pickle import load
from random import shuffle
from collections import Counter


# Let Keras know that we are using tensorflow as our backend engine
os.environ["KERAS_BACKEND"] = "tensorflow"

# To make sure that we can reproduce the experiment and get the same results
# np.random.seed(10)

# The dimension of our random noise vector.
random_dim = 100

# Load a neural network
# my_nn = load_model('new_nn.h5')
# mnist_nn = load_model('mnist_model.h5')

class Generator:
    def __init__(self, g = None):
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
        self.__init__()

class GAN:
    def __init__(self, random_dim=100, x_train=None, x_test=None, discriminator=Discriminator(),
                 generator=Generator()):
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
            attack=False):
        # Get the training and testing data
        x_train, y_train, x_test, y_test = 0, 0, 0, 0
        if self.curr_x_train is not None:
            x_train, x_test = self.curr_x_train, self.curr_x_test
            x_train = (x_train.astype(np.float32) - 127.5)/127.5
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] ** 2)
        else:
            x_train, y_train, x_test, y_test = load_minst_data()
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
                # Get a random set of input noise and images
                noise = np.random.normal(0, 1, size=[batch_size, random_dim])
                image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
                image_batch = image_batch.reshape(128, 784)  # remove later

                # Generate fake MNIST images
                generated_images = self.G.predict(noise)

                # # If we want to pass noise to the discriminator
                # new_noise = np.random.normal(0, 1, size=[batch_size, 784])

                X = np.concatenate([image_batch, generated_images])
                y_dis = np.zeros(2*batch_size)

                # # -----------------------------------------------------#
                # # Experience replay algorithm(still in testing)
                # # Comment out if you intend to use normal gan
                # # Create an interval larger than 1
                interval = 2
                if e % interval == 0:
                    # Create some noise
                    noise = np.random.normal(0, 1, size=(batch_size, 784))[:64]
                    # get the two most recent generations and reshape
                    x = np.array(old_imgs[-1:]).reshape(200, 784)
                    # shuffle the array 5 times
                    for _ in range(5):
                        shuffle(x)

                    # Randomly select images to use.
                    experience_rep = np.array([img for img in x[:64]] + list(noise))


                    # # Combine recently generated images and old ones
                    # gen_images = np.array(list(generated_images)[:64] + \
                    #             list(experience_rep))

                    # Concatenate the fake images and real ones
                    X = np.concatenate([image_batch, generated_images,
                                experience_rep])

                    y_dis = np.zeros(3*batch_size)

                # -----------------------------------------------------#

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
            # Plots the images if plot is set to True(default)
            # Can add an extra condition e.g. if id == 10
            possible_morphs = []
            if plot:
                possible_morphs = plot_generated_images(id, self.G)
            if possible_morphs is not None:
                for i in range(len(possible_morphs)):
                    morphed.append(possible_morphs[i])

            # ----------------------------------------------- #
            # Counting the images using a neural network
            # # View the count
            # if id % 1 != 0:
            #     print(get_count(self.G.predict(noise)))

            # ----------------------------------------------- #

            # Append recent generations to old images list
            # Realized the bug could be that they are in different formats
            # Trying out sth new. Will style up the code
            imgs = language_getter.produce_language(self.G, n=2).reshape(200,784)
            imgs = (imgs.astype(np.float32) - 127.5)/127.5
            old_imgs.append(imgs)
            # Increase id
            id += 1
        # print('This is how many morphed images were found')
        # print(len(morphed))
        # if len(morphed) > 100:
        #     plot_morphs(np.array(morphed[:100]))
        # else:
        #     plot_morphs(np.array(morphed))

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
    count_and_morphs = get_count(generated_images.reshape(examples, 784), id)
    # print(count_and_morphs[0])
    if count_and_morphs[1] != []:
        return [generated_images[x] for x in count_and_morphs[1]]

def plot_morphs(morphs, dim=(10, 10), figsize=(10,10)):
    ''' Plots the morphed images '''
    for i in range(morphs.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(morphs[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    newfile = filename = 'morphed'
    copy = 1
    while os.path.exists(newfile + '.png'):
        newfile = filename + '(' + str(copy) + ')'
        copy += 1
    plt.savefig(newfile)
    plt.close('all')


def get_count(data, id):
    ''' Uses the neural network to get counts '''
    pred = my_nn.predict(data)
    morphs = []
    # We want a greater quality of images
    if id > 15:
        for i in range(len(pred)):
            # Most of the time, the neural network will always give a prediction
            # that is > 0.9. Whenever the converse happens, we can assume there
            # is an element of morphing. Am therefore using a range of .4
            possible_morphs = [x for x in pred[i] if 0.4 <= x <= 0.6]
            if possible_morphs != []:
                # appends the image number
                morphs.append(i)
    # To see distribution of images, index the first element
    # To see if there were any morphs, index the second one
    return [Counter(['BCDEFGHIJK'[list(x).index(max(x))] for x in pred]),
            morphs]

# #
# if __name__ == '__main__':
# # #     # GAN().train(epochs=20)
#     vals = np.array(load(open('updated_lang_for_gan.txt', 'rb'))[:60000])
#     gen, disc = Generator(), Discriminator()
#     my_gan = GAN(x_train=vals, generator=gen, discriminator=disc)
#     my_gan.train(epochs=20)
#
#     # gen2, gen3, disc2 = Generator(), Generator(), Discriminator()
#     ganny1 = GAN(x_train=vals)
#     ganny2 = GAN(x_train=vals)
#     ganny2.G = ganny1.G
#     for _ in range(10):
#         ganny1.train()
#         ganny2.train()


    # gen2, disc2 = Generator(), Discriminator()
    # my_gan2 = GAN(x_train=language_getter.produce_language(my_gan.G),
    #               generator=gen2, discriminator=disc2)
    # my_gan2.train(epochs=20)
    # gen3, disc3 = Generator(), Discriminator()
    # my_gan3 = GAN(x_train=language_getter.produce_language(my_gan2.G),
    #               generator=gen3, discriminator=disc3)
    # my_gan3.train(epochs=20)

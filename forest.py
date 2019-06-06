''' A graph implementation of a forest by Ian Nduhiu, Kenny Talarico and
    Dave Perkins. Created on May 31, 2019. '''

import gan
import tree
from random import choice
import numpy as np

random_dim = 100

class Forest:
    def __init__(self):
        self.dekutree = tree.Tree()
        self.dekutree.brain.train()
        self.trees = [[None, None, None, None, None],
                      [None, None, None, None, None],
                      [None, None, self.dekutree, None, None],
                      [None, None, None, None, None],
                      [None, None, None, None, None]]

def write_to_file(np_array):
    ''' Writes a numpy array to a file '''
    np_array.tofile('language_L.txt')

def retrieve_from_file():
    ''' Retrieves numpy array from a file '''
    return np.fromfile('filename.txt')

def main():
    # gancontrol = gan.GAN(random_dim) #control
    # gen = gan.Generator()
    # disca = gan.Discriminator()
    # discb = gan.Discriminator()
    # testgans = [gan.GAN(random_dim, generator=gen, discriminator=disca),
    #             gan.GAN(random_dim, generator=gen, discriminator=discb)]
    #
    # epochs = 15
    #
    # gancontrol.train(epochs=epochs)
    #
    # for i in range(1, epochs+1):
    #     choice(testgans).train(id=i)

    # tree1 = tree.Tree()
    # tree2 = tree.Tree()
    #
    # gan1 = gan.GAN(random_dim, generator=tree1.generator, discriminator=tree2.discriminator)
    #
    # for _ in range(1):
    #     gan1.train()
    #     tree1.brain.train(plot=False)
    #     tree2.brain.train(plot=False)
    #
    # tree3 = tree.Tree()
    #
    # gan2 = gan.GAN(random_dim, generator=tree3.generator, discriminator=tree1.discriminator)
    #
    # gan2.train()

    gen1 = gan.Generator()
    gen2 = gan.Generator()
    disc1 = gan.Discriminator()
    disc2 = gan.Discriminator()

    gan1 = gan.GAN(random_dim, generator=gen1, discriminator=disc1)
    gen_images_1 = np.array(gan1.train(epochs=5))
    # gen_images_1 = gen_images_1.reshape(gen_images_1.shape[0] * \
                # gen_images_1.shape[1], gen_images_1.shape[2], gen_images_1.shape[3])

    # print(gen_images_1.shape)

    # gan2 = gan.GAN(random_dim, generator=gen2, discriminator=disc1)
    # gen_images_2 = gan2.train()

    # If you wish to write any of the generated images to a file
    # write_to_file(gen_images_1)

    # # If you want to retrieve generated images
    # my_array = retrieve_from_file()


if __name__ == '__main__':
    main()

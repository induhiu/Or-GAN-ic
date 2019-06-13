""" CREATING A LANGUAGE WITH COLLABORATIVE NEURAL NETWORKS
    Ian Nduhiu and Kenny Talarico, advisor David Perkins
    Hamilton College
    Summer 2019 """

import numpy as np
# import forest
import gan
from pickle import load
from language_getter import produce_language


def main():
    gen1 = gan.Generator()
    gen2 = gan.Generator()
    gen3 = gan.Generator()
    disc = gan.Discriminator()

    gan.GAN(generator=gen1, discriminator=disc, x_train=np.array(load(open('lang_for_gan.txt', 'rb'))[:60000])).train(epochs=20)
    disc.reset()
    gan.GAN(generator=gen2, discriminator=disc, x_train=produce_language(gen1.G)).train(epochs=15)
    disc.reset()
    gan.GAN(generator=gen3, discriminator=disc, x_train=produce_language(gen2.G)).train(epochs=10)




if __name__ == "__main__":
    main()

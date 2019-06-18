""" CREATING A LANGUAGE WITH COLLABORATIVE NEURAL NETWORKS
    Ian Nduhiu and Kenny Talarico, advisor David Perkins
    Hamilton College
    Summer 2019 """

import numpy as np
from gan import Generator
from gan import Discriminator
from gan import GAN
from pickle import load
from language_getter import produce_language
from tensorflow.keras.models import load_model
from collections import Counter

def main():
    # gen1 = Generator()
    # gen2 = Generator()
    # gen3 = Generator()
    #
    # dekug = load_model('./saveddekus/dekugen25.h5')
    # GAN(generator=gen1, x_train=produce_language(dekug)).train(epochs=20)
    # GAN(generator=gen2, x_train=produce_language(gen1.G)).train(epochs=20)
    # my_nn = load_model('nn.h5')
    # dekug = load_model('./saveddekus/dekugen20.h5')
    # lang = produce_language(dekug)
    # print(lang.shape)
    # pred = my_nn.predict(lang)
    # my_counter = Counter('ABCDEFGHIJ'[list(x).index(max(x))] for x in pred)
    # print(my_counter)
    GAN(x_train=np.array(pickle.load(open('lang_for_gan.txt', 'rb'))[:60000]))

if __name__ == "__main__":
    main()

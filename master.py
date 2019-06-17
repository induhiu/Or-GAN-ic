""" CREATING A LANGUAGE WITH COLLABORATIVE NEURAL NETWORKS
    Ian Nduhiu and Kenny Talarico, advisor David Perkins
    Hamilton College
    Summer 2019 """

import numpy as np
# import forest
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
    # dekug = load_model('./saveddekus/dekugen5.h5')
    # GAN(generator=gen1, x_train=produce_language(dekug)).train(epochs=5)
    # GAN(generator=gen2, x_train=produce_language(gen1.G)).train(epochs=5)
    # GAN(generator=gen3, x_train=produce_language(gen2.G)).train(epochs=5)
    my_nn = load_model('nn.h5')
    dekug = load_model('./saveddekus/dekugen20.h5')
    lang = produce_language(dekug)
    print(lang.shape)
    pred = my_nn.predict(lang)  # contains arrays with shape (10, 1)
    print(pred)
    my_counter = Counter('ABCDEFGHIJ'[list(x).index(max(x))] for x in pred)
    print(my_counter)

if __name__ == "__main__":
    main()

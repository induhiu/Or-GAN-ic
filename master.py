""" CREATING A LANGUAGE WITH COLLABORATIVE NEURAL NETWORKS
    Ian Nduhiu and Kenny Talarico, advisor David Perkins
    Hamilton College
    Summer 2019 """

import numpy as np
# import forest
import gan
from pickle import load
from language_getter import produce_language
from tensorflow.keras.models import load_model



def main():
    # gen1 = gan.Generator()
    # # gen2 = gan.Generator()
    # # gen3 = gan.Generator()
    # disc = gan.Discriminator()

    gansington = gan.GAN(x_train=np.array(load(open('lang_for_gan.txt', 'rb'))[:60000]))
    for i in range(1, 41):
        gansington.train(id=i)
        if i % 5 == 0 and i != 30:
            gansington.G.save('dekugen' + str(i) + '.h5')

    # disc.reset()
    # gan.GAN(generator=gen2, discriminator=disc, x_train=produce_language(gen1.G)).train(epochs=15)
    # disc.reset()
    # gan.GAN(generator=gen3, discriminator=disc, x_train=produce_language(gen2.G)).train(epochs=10)

    # gen = load_model('dekugen30.h5')
    # gen1 = gan.Generator(g=gen)
    # gan.GAN(x_train=produce_language(gen1.G)).train(epochs=10)



if __name__ == "__main__":
    main()

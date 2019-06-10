''' Test case two. Switching up the gans '''

import numpy as np
import gan

def slicedata(dic, size):
    vals = list(dic.values())
    li1 = vals[:size]
    li2 = vals[size:]
    return (li1, li2)

def main():
    xtrain, xtest = slicedata(np.load('imgarys.npz'), 60000)
    xtrain = np.array(xtrain)
    xtest = np.array(xtest)
    # print(xtrain)

    gen1 = gan.Generator()
    gen2 = gan.Generator()
    disc1 = gan.Discriminator()
    ganny = gan.GAN(generator=gen1, discriminator=disc1, x_train=xtrain, x_test=xtest)
    ganny2 = gan.GAN(generator=gen2, discriminator=disc1, x_train=xtrain, x_test=xtest)

    for i in range(10):
        ganny.train(id=i)
        ganny2.train(id=i)

    # gan.plot_generated_images(10, gen1)
    # gan.plot_generated_images(10, gen2)


if __name__ == "__main__":
    main()

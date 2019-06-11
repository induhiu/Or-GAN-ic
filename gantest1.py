import gan
import numpy as np

def slicedata(dic, size):
    vals = list(dic.values())
    li1 = vals[:size]
    li2 = vals[size:]
    return (li1, li2)

def main():
    xtrain, xtest = slicedata(np.load('imgarys.npz'), 60000)
    xtrain = np.array(xtrain)
    xtest = np.array(xtest)

    gen1 = gan.Generator()
    gen2 = gan.Generator()
    disc = gan.Discriminator()

    gan1 = gan.GAN(generator=gen1, discriminator=disc, x_train=xtrain, x_test=xtest)
    gan2 = gan.GAN(generator=gen2, discriminator=disc, x_train=xtrain, x_test=xtest)

    gan1.train(10)
    gan2.train(10)

if __name__ == '__main__':
    main()

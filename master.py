""" CREATING A LANGUAGE WITH COLLABORATIVE NEURAL NETWORKS
    Ian Nduhiu and Kenny Talarico, advisor David Perkins
    Hamilton College
    Summer 2019 """

import numpy as np
# import forest
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
    print(xtrain.shape)
    print(xtest.shape)

    ganny = gan.GAN(100, x_train=xtrain, x_test=xtest)
    ganny.train(10)

if __name__ == "__main__":
    main()

""" CREATING A LANGUAGE WITH COLLABORATIVE NEURAL NETWORKS
    Ian Nduhiu and Kenny Talarico, advisor David Perkins
    Hamilton College
    Summer 2019 """

import numpy as np
# import forest
# import gan
import pickle

my_dict = np.load('imgarys.npz')
#
# def slicedata(dic, size):
#     vals = list(dic.values())
#     li1 = vals[:size]
#     li2 = vals[size:]
#     return (li1, li2)

def sliced_key_and_val(key):
    ''' Returns a list of the value and its label '''
    return [my_dict[key], key[0]]

def main():
    # xtrain, xtest = slicedata(np.load('imgarys.npz'), 60000)
    # xtrain = np.array(xtrain)
    # xtest = np.array(xtest)
    # # print(xtrain)
    #
    # ganny = gan.GAN(100, x_train=xtrain, x_test=xtest)
    # ganny.train(10)
    # b = pickle.load(open('lang.txt', 'rb'))

    # print(len(b))
    results = list(map(sliced_key_and_val, list(my_dict.keys())))
    with open('lang_for_nn.txt', 'wb') as file:
        pickle.dump(results, file)
    print(results[0])
    print(len(results))


if __name__ == "__main__":
    main()

"""
CREATING A LANGUAGE WITH COLLABORATIVE NEURAL NETWORKS
Ian Nduhiu and Kenny Talarico, advisor David Perkins
Hamilton College
Summer 2019
"""

import numpy as np

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)


class Model():

    def __init__(self):
        pass

class Generator(Model):

    def __init__(self, in):
        Model.__init__()
        self.size = in.shape
        self.weight = np.random.rand(in.shape[1], in.shape[0])
        self.out = np.zeros(in.shape)

    def generate(din=None):
        # din is the discriminator input. If din is none this means that this is
        # the first time that we're calling generate and so the ouput should be
        # randomized. Otherwise, do something with din. We'll figure that out later.

        self.out = (np.random.rand(self.size[1], self.size[0]) if din is None else self.something(din))


class Discriminator(Model):

    def __init__(self, out):
        Model.__init__()
        self.out = out

    def discriminate(in):
        pass


def main():
    # test input and output
    samplein = np.array([[0,0,1],
                         [0,1,1],
                         [1,0,1],
                         [1,1,1]])
    sampleout = np.array([[0],[1],[1],[0]])
    sampleout = sampleout.reshape(4, 1)

    # Here through the for loop should always be here
    G = Generator(samplein)
    D = Discriminator(sampleout)

    G.generate()
    D.discriminate(G.out)

    # change as needed
    for _ in range(1):
        G.generate(D.out)
        D.discriminate(G.out)


if __name__ == "__main__":
    main()

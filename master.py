"""
CREATING A LANGUAGE WITH COLLABORATIVE NEURAL NETWORKS
Ian Nduhiu and Kenny Talarico, advisor David Perkins
Hamilton College
Summer 2019
"""

import numpy as np

class Model():

    def __init__(self, in=None, out=None):
        self.in = in
        # tempout holds the changing output
        self.tempout = out
        self.out = out



class Generator(Model):

    def __init__(self):
        Model.__init__()

class Discriminator(Model):

    def __init__(self):
        Model.__init__()




def main():
    samplein = np.array([[0,0,1],
                         [0,1,1],
                         [1,0,1],
                         [1,1,1]])
    G = Generator(samplein)
    D = Discriminator()

    G.generate()
    D.discriminate(G.out)

    # change as needed
    for _ in range(1):
        G.generate(D.out)
        D.discriminate(G.out)


if __name__ == "__main__":
    main()

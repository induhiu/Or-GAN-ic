""" CREATING A LANGUAGE WITH COLLABORATIVE NEURAL NETWORKS
Ian Nduhiu and Kenny Talarico, advisor David Perkins
Hamilton College
Summer 2019 """

import numpy as np
import mom
import gan
import wordidentifier as wi

alphabet = ['a', 'b', 'c', 'd', 'e']
signs = ['kenny', 'ian', 'dave', 'talarico', 'nduhiu', 'perkins']

def main():
    m = mom.Mom(alphabet, signs, wordlength=8, familysize=10)
    m.output('momwords.txt')
    w = wi.WordIdentifier(m.output())
    w.output()
    # xtrain = np.array(list(w.dict.values()))
    # ytrain = np.array(list(w.dict.keys())).reshape((len(w.dict), 1))

if __name__ == "__main__":
    main()

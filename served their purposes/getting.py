import numpy as np
import pickle

def main():
    ''' Main function '''
    dataset = []
    vals = np.load('imgarys.npz')
    for key in vals:
        dataset.append([vals[key], key[0]])
    with open('updated_lang_for_nn.txt', 'wb') as file:
        pickle.dump(dataset, file)

if __name__ == '__main__':
    main()

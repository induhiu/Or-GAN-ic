"""
Implementation of the "mom" neural networks using Karas.
Kenny Talarico and Ian Nduhiu, May 2019
"""

import mom
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def trainMoms(mum1, mum2):

    model = Sequential()
    model.add(Dense(mum1.size, activation='relu', input_shape=(1,)))
    model.add(Dense(mum1.size // 2, activation='relu'))
    model.add(Dense(len(mum1.signs), activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(np.array(list(mum1.dictionary.keys())).reshape(len(mum1.signs), 1))

    model.fit(np.array(list(mum1.dictionary.keys())).reshape(len(mum1.signs), 1),
              np.array(list(mum1.dictionary.values())).reshape(len(mum1.signs), 1),
              epochs=1, batch_size=1, verbose=1)

    ran = random.choice(list(mum1.dictionary.keys()))
    pred = model.predict(ran)
    print(pred)

def convertToBase(n, base, alphabet):
    """ Adapted from https://interactivepython.org/runestone/static/pythonds/Recursion/
        pythondsConvertinganIntegertoaStringinAnyBase.html """
    if n < base:
        return alphabet[n]
    else:
        return convertToBase(n // base, base, alphabet) + alphabet[n % base]

def main():
    # Change to fill array with any other values.
    # There can an arbitrarily large alphabet and up to len(alphabet)^2 signs.
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f']
    signs = ['kenny', 'ian', 'dave', 'decker']
    #
    mum1 = mom.Mom(alphabet, signs)
    mum2 = mom.Mom(alphabet, signs)

    trainMoms(mum1, mum2)
    print(mom.Mom(alphabet, signs))



if __name__ == '__main__':
    main()

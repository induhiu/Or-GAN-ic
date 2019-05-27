"""
Implementation of the "mom" neural networks using Karas.
Kenny Talarico and Ian Nduhiu, May 2019
"""

import mom
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def trainMoms(mum1, mum2):

    model = Sequential()
    model.add(Dense(mum1.size, activation='relu', input_dim=9))
    model.add(Dense(mum1.size + 5, activation='relu'))
    model.add(Dense(len(mum1.signs), activation='softmax'))  # was softmax

    x_train = np.empty((len(mum1.signs), len(list(mum1.dictionary.keys())[0])))
    prettierX = np.empty((len(mum1.signs), len(list(mum1.dictionary.keys())[0])), dtype='str')
    # x_train = np.array(list(mum1.dictionary.keys())).reshape(len(mum1.signs), 1)
    for word in range(len(mum1.dictionary.keys())):
        for letter in range(len(list(mum1.dictionary.keys())[0])):
            x_train[word][letter] = ord(list(mum1.dictionary.keys())[word][letter])
            prettierX[word][letter] = list(mum1.dictionary.keys())[word][letter]
    # print(x_train)


    a = np.array(list(mum1.dictionary.values()))
    y_train = a.reshape(len(mum1.signs), 1)

    # y_train = to_categorical(b)
    y_range = np.arange(0, len(y_train)).reshape(len(mum1.signs), 1)
    # print(y_range)

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_range, epochs=100, batch_size=1)

    ran = random.choice(list(mum1.dictionary.keys()))
    # for letter in range(len(ran)):
    #
    # pred = model.predict(np.array([ran]))

    # print(np.array([[ran[letter] for letter in range(len(ran))]]))

    pred = model.predict(np.array(([[ord(ran[letter]) for letter in range(len(ran))]])))
    print("List of fake words:", '\n', prettierX)
    print("List of real words:", '\n', y_train)
    print("Random word:", ran)
    print("My prediction:", pred)

def main():
    # Change to fill array with any other values.
    # There can an arbitrarily large alphabet and up to len(alphabet)^2 signs.
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f']
    # signs = ['kenny', 'ian', 'dave', 'decker']
    signs = [
    mum1 = mom.Mom(alphabet, signs)
    mum2 = mom.Mom(alphabet, signs)

    print(mum1)

    # trainMoms(mum1, mum2)


if __name__ == '__main__':
    main()

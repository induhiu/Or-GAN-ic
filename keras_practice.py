# ''' A very basic Keras Neural Network implementation '''
# # Import pandas
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# # Import `train_test_split` from `sklearn.model_selection`
# from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.layers import Dense
#
# # Read in white wine data
# white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
#
# # Read in red wine data
# red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')
#
# # Add `type` column to `red` with value 1
# red['type'] = 1
#
# # Add `type` column to `white` with value 0
# white['type'] = 0
#
# # Append `white` to `red`
# wines = red.append(white, ignore_index=True)
#
# # Specify the data
# X=wines.ix[:,0:11]
#
# # Specify the target labels and flatten the array
# y= np.ravel(wines.type)
#
# # Split the data up in train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#
# # Define the scaler
# scaler = StandardScaler().fit(X_train)
# # Scale the train set
# X_train = scaler.transform(X_train)
# # Scale the test set
# X_test = scaler.transform(X_test)
#
# # Initialize the constructor
# model = Sequential()
# # Add an input layer
# model.add(Dense(12, activation='relu', input_shape=(11,)))
# # Add one hidden layer
# model.add(Dense(8, activation='relu'))
# # Add an output layer
# model.add(Dense(1, activation='sigmoid'))
#
# # # Model output shape
# # model.output_shape
# # # Model summary
# # model.summary()
# # # Model config
# # model.get_config()
# # # List all weight tensors
# # model.get_weights()
#
# # Compile model. This is giving the model a loss function, optimizing function
# # and its metrics.
# model.compile(loss='binary_crossentropy',
#               optimizer='adam', metrics=['accuracy'])
#
# # Train the model
# model.fit(X_train, y_train,epochs=20, batch_size=1, verbose=1)

# ''' ------------------------------------------------------------ '''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Training data
data = np.random.random((1000,20)) # training
labels = np.random.randint(2,size=(1000,1)) # output

# Testing data
x_test = np.random.random((10, 20))
y_test = np.random.randint(2, size=(10, 1))

# print(labels)

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=20))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy', metrics=['accuracy'])
model.fit(data, labels,epochs=10,batch_size=32) # train
# if we want to evaluate the model's performance

# if we want to test the model on a batch
# x_batch = np.random.random((100,20))
# y_batch = np.random.randint(2, size=(100, 1))
pred = model.predict(x_test)
print(pred)


# '''--------------------------------------------------------------------------'''
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation
# from keras.optimizers import SGD
#
# # Generate dummy data
# import numpy as np
# x_train = np.random.random((1000, 20))
# y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
# x_test = np.random.random((100, 20))
# y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
#
# print(x_train)
# print(y_train)
#
# model = Sequential()
# # Dense(64) is a fully-connected layer with 64 hidden units.
# # in the first layer, you must specify the expected input data shape:
# # here, 20-dimensional vectors.
# model.add(Dense(64, activation='relu', input_dim=20))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))
#
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy',
#               optimizer=sgd,
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train,
#           epochs=20,
#           batch_size=128)
# pred = model.predict(x_test, y_test)

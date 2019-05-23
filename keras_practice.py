''' A very basic Keras Neural Network implementation '''
# Import pandas
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Read in white wine data
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')

# Read in red wine data
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

# Specify the data
X=wines.ix[:,0:11]

# Specify the target labels and flatten the array
y= np.ravel(wines.type)

# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Define the scaler
scaler = StandardScaler().fit(X_train)
# Scale the train set
X_train = scaler.transform(X_train)
# Scale the test set
X_test = scaler.transform(X_test)

# Initialize the constructor
model = Sequential()
# Add an input layer
model.add(Dense(12, activation='relu', input_shape=(11,)))
# Add one hidden layer
model.add(Dense(8, activation='relu'))
# Add an output layer
model.add(Dense(1, activation='sigmoid'))

# Model output shape
model.output_shape
# Model summary
model.summary()
# Model config
model.get_config()
# List all weight tensors
model.get_weights()
''' ------------------------------------------------------------ '''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Training data
data = np.random.random((1000,20)) # training
labels = np.random.randint(2,size=(1000,1)) # output

# Testing data
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))
# print(labels)

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=20))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy', metrics=['accuracy'])
model.fit(data, labels,epochs=10,batch_size=32) # train
predictions = model.predict(data)
# print(predictions)
# print(labels)

# if we want to evaluate the model's performance
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
print(loss_and_metrics)

# if we want to test the model on a batch
x_batch = np.random.random((100,20))
y_batch = np.random.randint(2, size=(100, 1))
results = test_on_batch(x_batch, y_batch)
print(results)

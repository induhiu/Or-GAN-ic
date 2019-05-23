''' A very basic Keras Neural Network implementation '''

# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense
#
# # Training data
# data = np.random.random((1000,20))
# labels = np.random.randint(2,size=(1000,1))
#
# # Testing data
# x_test = np.random.random((100, 20))
# y_test = np.random.randint(2, size=(100, 1))
# # print(labels)
#
# model = Sequential()
# model.add(Dense(32, activation='relu', input_dim=20))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(data, labels,epochs=10,batch_size=32)
# predictions = model.predict(data)
# print(predictions)
# print(labels)
#
# # if we want to evaluate the model's performance
# loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
# print(loss_and_metrics)

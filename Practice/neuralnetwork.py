# Documented by Kenny Talarico and Ian Nduhiu

import numpy as np

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        # self.input will be a NumPy array
        self.input      = x
        #shape returns a tuple
        #for array([[0], [1], [1], [0]]), shape returns (4, 1)
        #   np.random.rand creates an array of the specified with random
        #   values between 0 and 1
        self.weights1   = np.random.rand(self.input.shape[1],4)
        self.weights2   = np.random.rand(4,1)
        # NumPy array
        self.y          = y
        # NumPy array of zeros of given shape, doesn't really matter
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        dotp1 = np.dot(self.input, self.weights1)
        self.layer1 = sigmoid(dotp1)

        dotp2 = np.dot(self.layer1, self.weights2)
        self.output = sigmoid(dotp2)

    def backprop(self):
        # application of the chain rule to find derivative of the loss function
        #   with respect to weights2 and weights1

        # Uncomment to see error reduce
        # error = self.y - self.output
        # print(error.sum())

        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) *
                            sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output)
                            * sigmoid_derivative(self.output), self.weights2.T) *
                            sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


if __name__ == "__main__":
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])

    y = np.array([[0],[1],[1],[0]])
    y = y.reshape(4, 1)

    nn = NeuralNetwork(X,y)

    for i in range(1500):
        nn.feedforward()
        nn.backprop()

    print(nn.output)

#CITATION: https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

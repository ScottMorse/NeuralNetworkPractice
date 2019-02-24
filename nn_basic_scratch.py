import numpy as np
from pprint import pprint

class NeuralNetwork:

    def __init__(self):
        #set seed for re-running
        # np.random.seed(1)
        # 3 X 1 array of numbers between -1 and 1 (random range 0-2 minus 1)
        self.synaptic_weights = 2 * np.random.random((3,1)) - 1

    # S curve
    # The range of this function is only between 0 and 1, should match range of output desired (?)
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 / x)
    
    def train(self, inputs, outputs, iterations):
        for iteration in range(iterations):
            # uses current weights to find position on Sigmoid curve
            output = self.think(inputs)

            # compares difference between guess and real output
            err = outputs - output

            # adjusts for error using Sigmoid derivative
            adj = np.dot(inputs.T, err * self.sigmoid_derivative(output))

            # adjusts weights
            self.synaptic_weights += adj
        
    def think(self, inputs):
        inputs = inputs.astype(float)
        # input matrix multiplied by synaptic weight matrix and placed on Sigmoid curve
        return self.sigmoid(np.dot(inputs, self.synaptic_weights))

if __name__ == "__main__":

    nn = NeuralNetwork()

    print(f"Random Synaptic Weights:\n{nn.synaptic_weights}")
    
    train_inputs = np.array([
        [0,0,1],
        [1,1,1],
        [1,0,1],
        [0,1,1]
    ])

    # transposed for convenience to make them vertical (and match input as table)
    train_outputs = np.array([[0,1,1,0]]).T

    nn.train(train_inputs,train_outputs,10000)

    # probably has large number for first weight because first num of input is what determines?
    print(f"Post-training Weights:\n{nn.synaptic_weights}")

    test_input = np.array([1,0,0])

    print(f"Test result: {nn.think(test_input)}")

    # Should be close to 1. Good job computer



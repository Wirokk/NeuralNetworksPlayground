import numpy as np

class Netword(object): 

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weight = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]

    def sigmoid(self, z): 
        return 1/(1+np.exp(-z))
    
    def feedForward(self, a):
        for b, w in zip(self.biases, self.weight):
            a = np.dot(w, a) + b # Similar to perceptron output
            a = self.sigmoid(a) # Compression 0-1
        return a
    
    def SGD()
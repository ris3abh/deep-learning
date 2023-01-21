## testing the layers and the network class from layers.py

import numpy as np
from layers import *

##defining the network class to be used along with the layers for forward propagation

class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, dataIn):
        for layer in self.layers:
            dataIn = layer.forward(dataIn)
        return dataIn

    def backward(self, gradIn):
        for layer in reversed(self.layers):
            gradIn = layer.backward(gradIn)
        return gradIn

    def gradient(self):
        for layer in reversed(self.layers):
            layer.gradient()


## Testing the network class and the layers classes

L1 = inputLayer(np.array([[1,2,3,4,5],[6,7,8,9,10]]))
L2 = tanHLayer(L1)
L3 = fullyConnectedLayer(5, 3)
L4 = tanHLayer(L3)
L5 = fullyConnectedLayer(3, 2)
L6 = tanHLayer(L5)

X = [L1, L2, L3, L4, L5, L6]
net = Network(X)

print(net.forward(np.array([[1,2,3,4,5],[6,7,8,9,10]])))



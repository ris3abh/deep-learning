from ..core.base import Layer
from ..utils.initializers import xavier_init, he_init, uniform_init
import numpy as np

class FullyConnectedLayer(Layer):
    def __init__(self, size_in, size_out, init_type="xavier"):
        """
        Initialize a fully connected layer.
        
        Args:
            size_in (int): Input size
            size_out (int): Output size
            init_type (str): Type of initialization to use ("xavier", "he", or "uniform")
        """
        super().__init__()
        
        # Initialize weights and biases based on initialization type
        if init_type == "xavier":
            self.weights, self.biases = xavier_init(size_in, size_out)
        elif init_type == "he":
            self.weights, self.biases = he_init(size_in, size_out)
        else:
            self.weights, self.biases = uniform_init(size_in, size_out)

        # Adam optimizer accumulators
        self.weights_s = 0
        self.weights_r = 0
        self.biases_s = 0
        self.biases_r = 0

        # Adam hyperparameters
        self.decay_1 = 0.9
        self.decay_2 = 0.999
        self.stability = 10e-8

    # Rest of the class implementation remains the same
    def getWeights(self):
        return self.weights
    
    def setWeights(self, weights):
        self.weights = weights
    
    def getBiases(self):
        return self.biases
    
    def setBiases(self, biases):
        self.biases = biases

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(np.dot(dataIn, self.weights) + self.biases)
        return self.getPrevOut()

    def gradient(self):
        return np.array([self.weights.T for _ in range(len(self.getPrevIn()))])
    
    def updateWeights(self, gradIn, epoch, learning_rate=0.0001):
        # Calculate gradients for weights
        dJdw = (self.getPrevIn().T @ gradIn) / gradIn.shape[0]
        
        # Update momentum and RMSprop accumulators for weights
        self.weights_s = self.decay_1 * self.weights_s + (1 - self.decay_1) * dJdw
        self.weights_r = self.decay_2 * self.weights_r + (1 - self.decay_2) * dJdw * dJdw
        
        # Bias correction
        weights_update = (self.weights_s/(1-self.decay_1**(epoch+1))) / (
            np.sqrt(self.weights_r/(1-self.decay_2**(epoch+1))) + self.stability
        )
        
        # Calculate gradients for biases
        dJdb = np.sum(gradIn, axis=0)/gradIn.shape[0]
        
        # Update momentum and RMSprop accumulators for biases
        self.biases_s = self.decay_1 * self.biases_s + (1 - self.decay_1) * dJdb
        self.biases_r = self.decay_2 * self.biases_r + (1 - self.decay_2) * dJdb * dJdb
        
        # Bias correction
        biases_update = (self.biases_s/(1-self.decay_1**(epoch+1))) / (
            np.sqrt(self.biases_r/(1-self.decay_2**(epoch+1))) + self.stability
        )
        
        # Update weights and biases
        self.setWeights(self.getWeights() - learning_rate * weights_update)
        self.setBiases(self.getBiases() - learning_rate * biases_update)
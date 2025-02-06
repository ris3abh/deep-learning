# layers/regularization.py

from ..core.base import Layer
import numpy as np

class DropoutLayer(Layer):
    def __init__(self, keep_prob):
        """
        Initialize a dropout layer.
        
        Args:
            keep_prob (float): Probability of keeping a neuron active (between 0 and 1)
        """
        super().__init__()
        self.keep_prob = keep_prob
        self.dropOutKey = None

    def forward(self, dataIn, test=False, epoch=1):
        """
        Forward pass for dropout layer.
        
        Args:
            dataIn: Input data
            test (bool): Whether in test mode (no dropout) or training mode
            epoch (int): Current epoch number for reproducibility
            
        Returns:
            Output data with dropout applied (or not, if in test mode)
        """
        self.setPrevIn(dataIn)
        
        if test:
            self.setPrevOut(dataIn)
            return dataIn
        else:
            # Set random seed for reproducibility
            np.random.seed(epoch)
            
            # Generate dropout mask
            self.dropOutKey = np.random.rand(*dataIn.shape) < self.keep_prob
            
            # Apply dropout and scale
            dataOut = np.multiply(dataIn, self.dropOutKey)
            dataOut = dataOut / self.keep_prob
            
            self.setPrevOut(dataOut)
            return dataOut

    def gradient(self):
        """
        Compute the gradient for the dropout layer.
        """
        tensor = np.ones_like(self.dropOutKey) / self.keep_prob
        return tensor

    def backward(self, gradIn):
        """
        Backward pass for dropout layer.
        
        Args:
            gradIn: Incoming gradient
            
        Returns:
            Gradient for the layer
        """
        gradOut = gradIn * self.gradient()
        return gradOut
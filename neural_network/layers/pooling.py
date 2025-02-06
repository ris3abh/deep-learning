# layers/pooling.py

from ..core.base import Layer
import numpy as np

class PoolingLayer(Layer):
    def __init__(self, size, stride=1):
        """
        Initialize a pooling layer.
        
        Args:
            size (int): The size of the pooling window
            stride (int): The stride of the pooling operation
        """
        super().__init__()
        self.size = size
        self.stride = stride

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(self.pool(dataIn))
        return self.getPrevOut()

    def pool(self, dataIn):
        """Apply pooling across all samples in the batch."""
        return np.array([self.pool3D(dataIn[i]) for i in range(len(dataIn))])

    def pool3D(self, dataIn):
        """Apply pooling across all channels in a single sample."""
        return np.array([self.pool2D(dataIn[i]) for i in range(len(dataIn))])
    
    def pool2D(self, dataIn):
        """Apply max pooling to a single channel."""
        dataHeight, dataWidth = dataIn.shape
        outputWidth = int(((dataWidth - self.size) / self.stride) + 1)
        outputHeight = int(((dataHeight - self.size) / self.stride) + 1)
        output = np.zeros((outputHeight, outputWidth))

        for y in range(outputHeight):
            for x in range(outputWidth):
                output[y, x] = np.max(
                    dataIn[y*self.stride:y*self.stride+self.size, 
                          x*self.stride:x*self.stride+self.size]
                )

        return output
    
    def gradient(self):
        pass

    def backward(self, gradIn):
        """Compute gradient for the pooling layer."""
        return np.array([self.backward3D(grad_i, data) 
                        for data, grad_i in zip(self.getPrevIn(), gradIn)])
 
    def backward3D(self, gradIn, prevIn):
        """Compute gradient for each channel in 3D input."""
        return np.array([self.backwardRow(data, grad_i) 
                        for data, grad_i in zip(prevIn, gradIn)])

    def backwardRow(self, data, gradIn):
        """Compute gradient for a single channel."""
        dataHeight = data.shape[0]
        dataWidth = data.shape[1]
        output = np.zeros((dataHeight, dataWidth))

        for y in range(gradIn.shape[1]):
            for x in range(gradIn.shape[0]):
                grid = data[y*self.stride:y*self.stride+self.size, 
                          x*self.stride:x*self.stride+self.size]
                maxLoc = np.unravel_index(np.argmax(grid), (self.size, self.size))
                output[y*self.stride+maxLoc[0], x*self.stride+maxLoc[1]] = gradIn[y, x]

        return output
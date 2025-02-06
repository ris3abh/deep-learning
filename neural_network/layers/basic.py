# layers/basic.py

from ..core.base import Layer
import numpy as np

class InputLayer(Layer):
    def __init__(self, dataIn):
        super().__init__()
        self.meanX = np.mean(dataIn, axis=0)
        self.stdX = np.std(dataIn, axis=0, ddof = 1)
        self.stdX[self.stdX == 0] = 1
    
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        zscored = (dataIn - self.meanX) / self.stdX
        self.setPrevOut(zscored)
        return zscored

    def gradient(self):
        pass

class LinearLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(dataIn)
        return dataIn

    def gradient(self):
        return np.identity(self.getPrevIn().shape[1])

    def backward(self, gradIn):
        return gradIn

class FlattenLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(self.flatten(dataIn))
        return self.getPrevOut()

    def flatten(self, dataIn):
        return np.array([dataIn[i].flatten() for i in range(len(dataIn))])
    
    def gradient(self):
        pass
 
    def backward(self, gradIn):
        return gradIn.reshape(self.getPrevIn().shape)
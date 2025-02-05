from ..core.base import Layer
import numpy as np

class DropoutLayer(Layer):
    def __init__(self, keep_prob):
        super().__init__()
        self.keep_prob = keep_prob

    def forward(self, dataIn, test=False, epoch=1):
        self.setPrevIn(dataIn)
        if test:
            self.setPrevOut(dataIn)
            return dataIn
        else:
            np.random.seed(epoch)
            self.dropOutKey = np.random.rand(*dataIn.shape) < self.keep_prob
            dataOut = np.multiply(dataIn, self.dropOutKey)
            dataOut = dataOut / self.keep_prob
            self.setPrevOut(dataOut)
            return dataOut

    def gradient(self):
        tensor = np.ones_like(self.dropOutKey) / self.keep_prob
        return tensor

    def backward(self, gradIn):
        gradOut = gradIn * self.gradient()
        return gradOut
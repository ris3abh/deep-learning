from ..core.base import Layer
import numpy as np

EPSILON = 1e-7

class ReluLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        dataOut = np.maximum(0, dataIn)
        self.setPrevOut(dataOut)
        return dataOut

    def gradient(self):
        grad = np.where(self.getPrevOut() > 0, 1, 0)
        return grad

    def backward(self, gradIn):
        gradOut = gradIn * self.gradient()
        return gradOut

class LogisticSigmoidLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(1 / (1 + np.exp(-dataIn)))
        return self.getPrevOut()
    
    def gradient(self):
        diag = self.getPrevOut() * (1 - self.getPrevOut()) + EPSILON
        return np.eye(len(self.getPrevOut()[0])) * diag[:, np.newaxis]

class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        max = np.max(dataIn, axis = 1)[:, np.newaxis]
        self.setPrevOut(np.exp(dataIn - max) / np.sum(np.exp(dataIn - max), axis=1, keepdims=True))
        return self.getPrevOut()

    def gradient(self):
        out = self.getPrevOut()
        tensor = np.empty((0, out.shape[1], out.shape[1]))
        for row in out:
            grad = -(row[:, np.newaxis])*row
            np.fill_diagonal(grad, row*(1-row))
            tensor = np.append(tensor, grad[np.newaxis], axis = 0)
        return tensor

class TanhLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        dataInCal = np.where(dataIn > 100, 99, dataIn)
        dataInCal = np.where(dataInCal < -100, -99, dataInCal)
        dataOut = (np.exp(dataInCal) - np.exp(-dataInCal)) / (np.exp(dataInCal) + np.exp(-dataInCal))
        self.setPrevOut(dataOut)
        return dataOut

    def gradient(self):
        a = .000000000000001
        tensor = (1 - self.getPrevOut()**2) + a
        tensor = np.array(tensor)
        return tensor

    def backward(self, gradIn):
        gradOut = gradIn * self.gradient()
        return gradOut
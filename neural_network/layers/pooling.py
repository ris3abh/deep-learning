from ..core.base import Layer
import numpy as np


class PoolingLayer(Layer):
    def __init__(self, size, stride=1):
        super().__init__()
        self.size = size
        self.stride = stride

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(self.pool(dataIn))
        return self.getPrevOut()

    def pool(self, dataIn):
        return np.array([self.pool3D(dataIn[i]) for i in range(len(dataIn))])

    def pool3D(self, dataIn):
        return np.array([self.pool2D(dataIn[i]) for i in range(len(dataIn))])

    def pool2D(self, dataIn):
        dataHeight, dataWidth = dataIn.shape
        outputWidth = int(((dataWidth - self.size) / self.stride) + 1)
        outputHeight = int(((dataHeight - self.size) / self.stride) + 1)
        output = np.zeros((outputHeight, outputWidth))

        for y in range(outputHeight):
            for x in range(outputWidth):
                output[y, x] = np.max(dataIn[y * self.stride:y * self.stride + self.size, x * self.stride:x * self.stride + self.size])

        return output

    def gradient(self):
        pass

    def backward(self, gradIn):
        return np.array([self.backward3D(grad_i, data) for data, grad_i in zip(self.getPrevIn(), gradIn)])

    def backward3D(self, gradIn, prevIn):
        return np.array([self.backwardRow(data, grad_i) for data, grad_i in zip(prevIn, gradIn)])

    def backwardRow(self, data, gradIn):
        dataHeight = data.shape[0]
        dataWidth = data.shape[1]
        output = np.zeros((dataHeight, dataWidth))

        for y in range(gradIn.shape[1]):
            for x in range(gradIn.shape[0]):
                grid = data[y * self.stride:y * self.stride + self.size, x * self.stride:x * self.stride + self.size]
                maxLoc = np.unravel_index(np.argmax(grid), (self.size, self.size))
                output[y * self.stride + maxLoc[0], x * self.stride + maxLoc[1]] = gradIn[y, x]

        return output
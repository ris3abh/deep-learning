# core/base.py

from abc import ABC, abstractmethod
import numpy as np

EPSILON = 1e-7

class Layer(ABC):
    def __init__(self):
        self.__prevIn = []
        self.__prevOut = []

    def setPrevIn(self,dataIn):
        self.__prevIn = dataIn

    def setPrevOut(self, out):
        self.__prevOut = out

    def getPrevIn(self):
        return self.__prevIn

    def getPrevOut(self):
        return self.__prevOut
 
    def backward(self, gradIn):
        grad = self.gradient()
        return np.array([np.dot(gradIn_i, grad_i) for gradIn_i, grad_i in zip(gradIn, grad)])

    @abstractmethod
    def forward(self,dataIn):
        pass

    @abstractmethod  
    def gradient(self):
        pass
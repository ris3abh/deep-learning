from abc import ABC, abstractmethod 
import numpy as np

class Layer (ABC) :
    def init (self): 
        self . prevIn = [] 
        self. prevOut=[]

    def setPrevIn(self ,dataIn): 
        self . prevIn = dataIn

    def setPrevOut( self , out ): 
        self . prevOut = out

    def getPrevIn( self ):
        return self . prevIn

    def getPrevOut( self ):
        return self . prevOut

    @abstractmethod
    def forward(self ,dataIn):
        pass

    @abstractmethod
    def gradient(self):
        pass

    @abstractmethod
    def backward( self , gradIn ):
        pass

class inputLayer(Layer):
    def __init__(self, dataIn):
        self.meanX = np.mean(dataIn, axis=0)
        self.stdX = np.std(dataIn, axis=0)
        self.stdX[self.stdX == 0] = 1

    def forward(self, dataIn):
        return (dataIn - self.meanX) / self.stdX


    def gradient(self):
        pass

    def backward(self,gradIn):
        pass

class LinearLayer(Layer):
    def __init__(self, inputSize, outputSize):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.W = np.random.randn(self.inputSize, self.outputSize) * np.sqrt(2 / self.inputSize)
        self.b = np.zeros(self.outputSize)

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        return np.dot(dataIn, self.W) + self.b

    def gradient(self):
        pass

    def backward(self, gradIn):
        pass

class ReLULayer(Layer):
    def __init__(self):
        pass

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        return np.maximum(dataIn, 0)

    def gradient(self):
        pass

    def backward(self, gradIn):
        pass

class LogisticSigmoidLayer(Layer):
    def __init__(self):
        pass

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        return 1 / (1 + np.exp(-dataIn))

    def gradient(self):
        pass

    def backward(self, gradIn):
        pass

class SoftmaxLayer(Layer):
    def __init__(self):
        pass

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        exps = np.exp(dataIn - np.max(dataIn))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def gradient(self):
        pass

    def backward(self, gradIn):
        pass

class TanhLayer(Layer):
    def __init__(self):
        pass

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        return np.tanh(dataIn)

    def gradient(self):
        pass

    def backward(self, gradIn):
        pass

class FullyConnectedLayer(Layer):
    def __init__(self, sizeIn, sizeOut):
        self.sizeIn = sizeIn
        self.sizeOut = sizeOut
        self.W = np.random.randn(self.sizeIn, self.sizeOut) * np.sqrt(2 / self.sizeIn)
        self.b = np.zeros(self.sizeOut)

    def getWeights(self):
        return self.W

    def getBias(self):
        return self.b

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        return np.dot(dataIn, self.W) + self.b

    def gradient(self):
        pass

    def backward(self, gradIn):
        pass

X = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])



#Input Layer
inputLayer = inputLayer(X)
# print(inputLayer.forward(X))

#Linear Layer
linearLayer = LinearLayer(4, 3)
# print(linearLayer.forward(X))

#ReLu Layer
reluLayer = ReLULayer()
# print(reluLayer.forward(X))

#Logistic Sigmoid Layer
logisticSigmoidLayer = LogisticSigmoidLayer()
# print(logisticSigmoidLayer.forward(X))

#Softmax Layer
softmaxLayer = SoftmaxLayer()
# print(softmaxLayer.forward(X))

#Tanh Layer
tanhLayer = TanhLayer()
# print(tanhLayer.forward(X))

#Fully Connected Layer
fullyConnectedLayer = FullyConnectedLayer(4, 2)
print(fullyConnectedLayer.forward(X))



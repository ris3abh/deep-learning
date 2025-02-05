import numpy as np

EPSILON = 1e-7

class SquaredError:
    def eval(self, Y, Yhat):
        return np.mean((Y - Yhat) ** 2)

    def gradient(self, Y, Yhat):
        return -2*(Y - Yhat)

class LogLoss:
    def eval(self, Y, Yhat):
        return np.mean(-(Y * np.log(Yhat + EPSILON) + (1 - Y) * np.log(1 - Yhat + EPSILON)))

    def gradient(self, Y, Yhat):
        return -((Y - Yhat) / (Yhat * (1 - Yhat) + EPSILON))

class CrossEntropy:
    def eval(self, Y, Yhat):
        return -np.mean(np.sum(Y * np.log(Yhat + EPSILON), axis = 1))
    
    def gradient(self, Y, Yhat):
        return -np.divide(Y, (Yhat + EPSILON))
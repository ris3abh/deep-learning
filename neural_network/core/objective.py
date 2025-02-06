# core/objective.py

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
    
class NegativeLikelihood():
    def eval(self, y , yhat):
        yhat = np.clip(yhat, EPSILON, 1 - EPSILON)
        return -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
        ## should return a single value

    def gradient(self, y, yhat):
        return -np.divide(y, yhat + EPSILON) + np.divide((1-y), (1-yhat + EPSILON))
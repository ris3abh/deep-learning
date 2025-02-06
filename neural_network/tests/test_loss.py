import numpy as np
import sys
import os
import pytest

# Ensure that the project root is on the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the objective functions.
from neural_network.core.objective import SquaredError, NegativeLikelihood, CrossEntropy

def test_squared_error_loss():
    # Instantiate the squared error loss function
    L1 = SquaredError()
    
    # Define test targets and predictions
    Y = np.array([[0], [1]])
    Yhat = np.array([[0.2], [0.3]])
    
    # Expected loss: mean((Y - Yhat)^2) = (0.04 + 0.49)/2 = 0.265
    expected_loss = 0.265
    loss = L1.eval(Y, Yhat)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-5, atol=1e-5)
    
    # Expected gradient: -2*(Y - Yhat) = -2*([[0-0.2], [1-0.3]]) = [[0.4], [-1.4]]
    expected_grad = np.array([[0.4], [-1.4]])
    grad = L1.gradient(Y, Yhat)
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-5, atol=1e-5)

def test_negative_likelihood():
    L2 = NegativeLikelihood()
    
    # For binary classification, assume:
    Y = np.array([[0], [1]])
    Yhat = np.array([[0.2], [0.3]])
    
    # Expected loss: mean(-log(1 - yhat) for y=0, and -log(yhat) for y=1)
    # Sample 1: -log(0.8); Sample 2: -log(0.3)
    expected_loss = (-np.log(0.8) - np.log(0.3)) / 2
    loss = L2.eval(Y, Yhat)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-5, atol=1e-5)
    
    # Expected gradient:
    # For sample 1 (y=0): gradient = 1/(1 - 0.2) = 1/0.8 = 1.25
    # For sample 2 (y=1): gradient = -1/0.3 ≈ -3.33333333
    expected_grad = np.array([[1.25], [-3.33333333]])
    grad = L2.gradient(Y, Yhat)
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-5, atol=1e-5)

def test_cross_entropy_loss():
    L3 = CrossEntropy()
    
    # For multi-class classification with one-hot encoded labels:
    Y = np.array([[0, 1, 0],
                  [0, 1, 0]])
    Yhat = np.array([[0.2, 0.2, 0.6],
                     [0.2, 0.7, 0.1]])
    
    # Expected loss per sample:
    # Sample 1: -log(0.2), Sample 2: -log(0.7)
    # Mean loss = (-log(0.2) - log(0.7)) / 2
    expected_loss = (-np.log(0.2) - np.log(0.7)) / 2
    loss = L3.eval(Y, Yhat)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-5, atol=1e-5)
    
    # Expected gradient: -Y / (Yhat + EPSILON)
    # For sample 1: for class 1, gradient = -1/0.2 = -5.0; for others, 0.
    # For sample 2: for class 1, gradient = -1/0.7 ≈ -1.42857143; for others, 0.
    expected_grad = np.array([[0, -5.0, 0],
                              [0, -1.42857143, 0]])
    grad = L3.gradient(Y, Yhat)
    np.testing.assert_allclose(grad, expected_grad, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__])
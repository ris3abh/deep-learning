import numpy as np
import sys
import os
import pytest

# Ensure that the project root is on the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from neural_network.layers.basic import InputLayer, LinearLayer
from neural_network.layers.activations import ReluLayer, SoftmaxLayer, LogisticSigmoidLayer, TanhLayer
from neural_network.layers.dense import FullyConnectedLayer

def test_forward_and_backward():
    # -------------------------------
    # Instantiate and configure layers
    # -------------------------------
    # Create a fully connected layer with 3 inputs and 2 outputs
    L6 = FullyConnectedLayer(3, 2)

    # Define custom weights and biases for the fully connected layer
    W = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])
    b = np.array([-1, 2])

    # Set the weights and biases of the fully connected layer
    L6.setWeights(W)
    L6.setBiases(b)

    # -------------------------------
    # Prepare input data
    # -------------------------------
    # Define the input data (H)
    H = np.array([[1, 2, 3],
                  [4, 5, 6]])

    # -------------------------------
    # Instantiate additional layers
    # -------------------------------
    # Create an input layer and several activation layers
    L0 = InputLayer(H)
    L1 = ReluLayer()
    L2 = SoftmaxLayer()
    L3 = LogisticSigmoidLayer()
    L4 = TanhLayer()
    L5 = LinearLayer()

    # Group the layers into a list for sequential processing.
    Layers = [L0, L1, L2, L3, L4, L5, L6]

    # -------------------------------
    # Forward Pass
    # -------------------------------
    H_forward = H  # Start with the input data.
    for layer in Layers:
        H_forward = layer.forward(H_forward)
    
    # The expected final output (based on your printed example output)
    expected_final = np.array([[3.72077899, 8.29437199],
                               [3.72077899, 8.29437199]])
    
    # Check that the final forward pass output is close to expected.
    np.testing.assert_allclose(H_forward, expected_final, rtol=1e-5, atol=1e-5)

    # -------------------------------
    # Backward Pass (Gradient Calculation)
    # -------------------------------
    grad = None
    for layer in reversed(Layers):
        grad = layer.gradient()
    
    # Typically, the input layer does not produce a gradient (or returns None),
    # so verify that the final gradient is None.
    assert grad is None

if __name__ == "__main__":
    pytest.main([__file__])
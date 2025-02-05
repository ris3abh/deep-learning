import numpy as np

def xavier_init(size_in, size_out):
    """Xavier/Glorot initialization."""
    bound = np.sqrt(6/(size_in + size_out))
    weights = np.random.uniform(-bound, bound, (size_out, size_in)).T
    biases = np.random.uniform(-bound, bound, (1, size_out))
    return weights, biases

def he_init(size_in, size_out):
    """He initialization."""
    std_dev1 = np.sqrt(2/size_in)
    std_dev2 = np.sqrt(2/1)
    weights = np.random.normal(0, std_dev1, (size_in, size_out))
    biases = np.random.normal(0, std_dev2, (1, size_out))
    return weights, biases

def uniform_init(size_in, size_out, scale=0.001):
    """Simple uniform initialization."""
    weights = np.random.uniform(-scale, scale, (size_out, size_in)).T
    biases = np.random.uniform(-scale, scale, (1, size_out))
    return weights, biases
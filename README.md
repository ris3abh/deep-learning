## Neural Network Architecture from Scratch

This repository is designed for practicing and implementing neural network architectures from scratch. It aims to provide a comprehensive understanding of how neural networks work by building them without relying on high-level frameworks like TensorFlow or PyTorch. The main objective is to create a baseline and learn about different architectures in Computer Vision, NLP, and other domains.

#### Table of Contents
1. Project Overview
2. Directory Structure
3. Getting Started
4. Usage
5. Contributing
6. License

#### 1. Project Overview
This repository contains implementations of various neural network components and architectures, including layers, activation functions, loss functions, and optimizers. The goal is to provide a hands-on learning experience for those interested in understanding the inner workings of neural networks.


#### 2. Directory Structure
The repository is organized into several directories, each serving a specific purpose:

```plaintext
neural_network/
│
├── core/
│   ├── __init__.py
│   ├── base.py           # Contains abstract Layer class
│   └── objective.py      # Contains all objective/loss functions
│
├── layers/
│   ├── __init__.py
│   ├── basic.py         # Basic layers (Input, Linear, Flatten)
│   ├── activations.py   # All activation functions (ReLU, Sigmoid, Tanh)
│   ├── dense.py         # FullyConnected layer
│   ├── convolution.py   # Conv2D, Conv3D layers
│   ├── pooling.py       # Pooling layers
│   └── regularization.py # Dropout and other regularization layers
│
├── architectures/
│   ├── __init__.py
│   ├── cnn.py           # CNN specific architectures
│   ├── gan.py           # GAN architectures
│   └── autoencoder.py   # Autoencoder architectures
│
├── utils/
│   ├── __init__.py
│   ├── initializers.py  # Weight initialization methods
│   └── optimizers.py    # Optimization algorithms (Adam, etc.)
│
└── examples/
    ├── cnn_example.py
    ├── gan_example.py
    └── autoencoder_example.py
```


#### 3. Getting Started
To get started with this repository, follow these steps:

3.1. Clone the Repository:

```bash
git clone https://github.com/ris3abh/deep-learning.git
cd neural_network
```

3.2. Install Dependencies:

This project primarily uses NumPy for numerical computations. You can install it using pip:
```bash
pip install numpy
```

3.3. Explore the Code:

Each directory contains specific components of a neural network. Start with the core directory to understand the base classes and move on to layers, architectures, and utils for more advanced implementations.

#### 4. Usage

4.1. Running Examples
The examples directory contains example scripts for different neural network architectures. You can run these examples to see how the components work together:
```bash
python examples/cnn_example.py
python examples/gan_example.py
python examples/autoencoder_example.py
```

4.1.1. Adding New Components
    To add new layers, activation functions, or architectures, follow the existing structure:

4.1.2. Create a New File:
    Add a new Python file in the appropriate directory (e.g., layers/new_layer.py).
      
4.1.2. Implement the Component:
    Ensure your new component follows the structure of existing classes. For example, a new layer should inherit from the Layer class in core/base.py.
    
4.1.3. Update __init__.py:
    Add an import statement for your new component in the corresponding __init__.py file to make it accessible.

4.1.4. Testing
To ensure your implementations work correctly, write unit tests for each component. You can use a testing framework like unittest or pytest.


#### 5. Contributing
Contributions are welcome! If you would like to contribute, please follow these steps:

5.1. Fork the Repository:
Fork this repository to your GitHub account.

5.2. Create a New Branch:
```bash
git checkout -b feature/new-feature
```
5.3. Make Your Changes:
Implement your new feature or fix.

5.4. Commit and Push:
```bash
git commit -m "Add new feature"
git push origin feature/new-feature
```
5.5. Create a Pull Request:
Open a pull request on the main repository to merge your changes.

#### 6. License
This project is licensed under the MIT License. See the LICENSE file for details.

from ..core.base import Layer
import numpy as np

class Conv2DLayer(Layer):
    def __init__(self, filters, kernel_size, stride=1, padding=0):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel = self.init_kernel()
        self.stride = stride
        self.padding = padding

        # Adam optimizer accumulators
        self.weights_s = 0
        self.weights_r = 0
        self.biases_s = 0
        self.biases_r = 0

        self.decay_1 = 0.9
        self.decay_2 = 0.999
        self.stability = 10e-8

    def init_kernel(self):
        bound = np.sqrt(6/(self.filters*self.kernel_size[0]*self.kernel_size[1]))
        return np.random.uniform(-bound, bound, (self.filters, self.kernel_size[0], self.kernel_size[1]))

    def getKernel(self):
        return self.kernel
    
    def setKernel(self, kernel):
        self.kernel = kernel
    
    def getPadding(self):
        return self.padding
        
    def setPadding(self, padding):
        self.padding = padding
    
    def getStride(self):
        return self.stride
    
    def setStride(self, stride):
        self.stride = stride

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(self.convolve(dataIn, self.kernel, self.padding, self.stride))
        return self.getPrevOut()

    def convolve(self, dataIn, kernel, padding, stride):
        return np.array([[self.convolve2D(dataIn_i, kernel_i, padding, stride) 
                         for kernel_i in kernel] for dataIn_i in dataIn])

    def convolve2D(self, dataIn, kernel, padding=0, stride=1):       
        kh, kw = kernel.shape
        dh, dw = dataIn.shape

        # Calculate output dimensions
        oh = (dh - kh + 2 * padding) // stride + 1
        ow = (dw - kw + 2 * padding) // stride + 1

        # Create padded data
        if padding != 0:
            pad_dims = ((padding, padding), (padding, padding))
            dataIn = np.pad(dataIn, pad_dims)

        # Create a view of the input data
        sh, sw = dataIn.strides
        shape = (oh, ow, kh, kw)
        strides = (sh * stride, sw * stride, sh, sw)
        data_view = np.lib.stride_tricks.as_strided(dataIn, shape=shape, strides=strides)

        # Apply convolution
        output = np.tensordot(data_view, kernel, axes=[(2, 3), (0, 1)])

        return output

    def gradient(self):
        return np.array([np.transpose(self.kernel, (0, 2, 1))]*len(self.getPrevIn()))

    def backward(self, gradIn):
        grad = self.gradient()
        return np.array([self.backward2D(grad_i, gradIn_i) 
                        for gradIn_i, grad_i in zip(gradIn, grad)])
        
    def backward2D(self, grad, gradIn):
        return np.array([self.convolve2D(np.pad(gradIn_i, self.kernel_size[0]-1, constant_values=0), grad_i) 
                        for gradIn_i, grad_i in zip(gradIn, grad)])

    def updateKernel(self, gradIn, epoch, learning_rate=0.0001):
        for gradIn_i in gradIn:
            dJdw = self.convolve2D(self.getPrevIn(), gradIn_i, padding=0, stride=1)
            self.weights_s = self.decay_1 * self.weights_s + (1 - self.decay_1) * dJdw
            self.weights_r = self.decay_2 * self.weights_r + (1 - self.decay_2) * dJdw * dJdw
            weights_update = (self.weights_s / (1 - self.decay_1**(epoch+1))) / (
                np.sqrt(self.weights_r / (1 - self.decay_2**(epoch+1))) + self.stability
            )
            self.setKernel(self.getKernel() - learning_rate * weights_update)

class Conv3DLayer(Conv2DLayer):
    def __init__(self, filters, kernel_size, stride=1, padding=0):
        super().__init__(filters, kernel_size, stride, padding)
    
    def convolve(self, dataIn, kernel, padding, stride):
        return np.array([self.convolve3D(dataIn_i, kernel, padding, stride) 
                        for dataIn_i in dataIn])
    
    def convolve3D(self, dataIn, kernel, padding, stride):
        arr = np.array([[self.convolve2D(dataIn_i, kernel_i, padding, stride) 
                        for kernel_i in kernel] for dataIn_i in dataIn])
        arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
        return arr

    def gradient(self):
        return np.array([self.gradient2D()]*len(self.getPrevIn()))

    def gradient2D(self):
        arr = np.array([np.transpose(self.kernel, (0, 2, 1))]*len(self.getPrevIn()[0]))
        return arr.reshape(-1, arr.shape[-2], arr.shape[-1])

    def updateKernel(self, gradIn, epoch, learning_rate=0.0001):
        for gradIn_i in gradIn:
            for dataIn_i in self.getPrevIn():
                self.updateKernel3D(gradIn_i, dataIn_i, epoch, learning_rate)

    def updateKernel3D(self, gradIn, dataIn, epoch, learning_rate=0.0001):
        for i in range(len(dataIn)):
            for j in range(self.filters):
                self.updateKernel2D(gradIn[i + j], dataIn[i], epoch, learning_rate)
"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np


class NeuralLayer:
    def __init__(self, in_features, out_features, weight_init="random"):
        """
        Fully connected linear layer
        """

        if weight_init == "random":
            self.W = np.random.randn(in_features, out_features) * 0.01

        elif weight_init == "xavier":
            self.W = np.random.randn(in_features, out_features) * np.sqrt(1.0 / in_features)
        
        elif weight_init == "zeros":
            self.W = np.zeros((in_features, out_features))
            
        else:
            raise ValueError("Unsupported weight initialization method")

        self.b = np.zeros((1, out_features))

        self.grad_W = None
        self.grad_b = None

    def forward(self, X):
        """
        X shape: (batch_size, in_features)
        """
        self.X = X
        return X @ self.W + self.b

    def backward(self, dZ):
        """
        dZ shape: (batch_size, out_features)
        """
        # Gradients
        self.grad_W = self.X.T @ dZ
        self.grad_b = np.sum(dZ, axis=0, keepdims=True)

        # Gradient to pass backward
        dX = dZ @ self.W.T
        return dX
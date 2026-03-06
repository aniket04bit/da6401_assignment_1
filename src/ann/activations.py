"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np


class ReLU:
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, dA):
        return dA * (self.X > 0)
    
class Sigmoid:
    def forward(self, X):
        self.out = 1 / (1 + np.exp(-X))
        return self.out

    def backward(self, dA):
        return dA * (self.out * (1 - self.out))
    
class Tanh:
    def forward(self, X):
        self.out = np.tanh(X)
        return self.out

    def backward(self, dA):
        return dA * (1 - self.out ** 2)
    
class Softmax:
    def forward(self, X):
        X_shifted = X - np.max(X, axis=1, keepdims=True)
        exp_vals = np.exp(X_shifted)
        self.out = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        return self.out
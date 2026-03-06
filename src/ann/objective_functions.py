"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np


class CrossEntropy:
    def forward(self, logits, y):
        """
        logits: (batch_size, num_classes)
        y: one-hot encoded labels (batch_size, num_classes)
        """

        # Numerical stability
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)

        exp_vals = np.exp(shifted_logits)
        self.probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

        self.y = y

        loss = -np.sum(y * np.log(self.probs + 1e-8)) / y.shape[0]
        return loss

    def backward(self):
        batch_size = self.y.shape[0]
        return (self.probs - self.y) / batch_size
    
class MSE:
    def forward(self, preds, y):
        """
        preds: model output
        y: true labels (one-hot)
        """
        self.preds = preds
        self.y = y
        loss = np.mean((preds - y) ** 2)
        return loss

    def backward(self):
        batch_size = self.y.shape[0]
        return 2 * (self.preds - self.y) / batch_size
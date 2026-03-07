"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np


class CrossEntropy:
    def forward(self, y_true, y_pred):

        if y_true.ndim == 1:
            y_true = np.eye(y_pred.shape[1])[y_true]

        shifted = y_pred - np.max(y_pred, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)

        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        self.y_true = y_true

        B = y_true.shape[0]

        loss = -np.sum(y_true * np.log(self.probs + 1e-12)) / B

        return loss

    def backward(self):

        B = self.y_true.shape[0]
        return (self.probs - self.y_true) / B
    
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
"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
import numpy as np
class SGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, layers):
        """
        layers: list of network layers
        """
        for layer in layers:
            # Only update layers that have weights
            if hasattr(layer, "W"):
                layer.W -= self.lr * layer.grad_W
                layer.b -= self.lr * layer.grad_b
                
class Momentum:
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.lr = learning_rate
        self.beta = beta
        self.velocities = {}

    def update(self, layers):

        for idx, layer in enumerate(layers):

            if hasattr(layer, "W"):

                if idx not in self.velocities:
                    self.velocities[idx] = {
                        "vW": np.zeros_like(layer.W),
                        "vB": np.zeros_like(layer.b)
                    }

                vW = self.velocities[idx]["vW"]
                vB = self.velocities[idx]["vB"]

                # Update velocity
                vW = self.beta * vW - self.lr * layer.grad_W
                vB = self.beta * vB - self.lr * layer.grad_b

                # Update weights
                layer.W += vW
                layer.b += vB

                self.velocities[idx]["vW"] = vW
                self.velocities[idx]["vB"] = vB
                
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = {}
        self.v = {}
        self.t = 0  # timestep

    def update(self, layers):

        self.t += 1

        for idx, layer in enumerate(layers):

            if hasattr(layer, "W"):

                if idx not in self.m:
                    self.m[idx] = {
                        "mW": np.zeros_like(layer.W),
                        "mB": np.zeros_like(layer.b)
                    }
                    self.v[idx] = {
                        "vW": np.zeros_like(layer.W),
                        "vB": np.zeros_like(layer.b)
                    }

                # Get gradients
                gW = layer.grad_W
                gB = layer.grad_b

                # Update first moment
                self.m[idx]["mW"] = self.beta1 * self.m[idx]["mW"] + (1 - self.beta1) * gW
                self.m[idx]["mB"] = self.beta1 * self.m[idx]["mB"] + (1 - self.beta1) * gB

                # Update second moment
                self.v[idx]["vW"] = self.beta2 * self.v[idx]["vW"] + (1 - self.beta2) * (gW ** 2)
                self.v[idx]["vB"] = self.beta2 * self.v[idx]["vB"] + (1 - self.beta2) * (gB ** 2)

                # Bias correction
                mW_hat = self.m[idx]["mW"] / (1 - self.beta1 ** self.t)
                mB_hat = self.m[idx]["mB"] / (1 - self.beta1 ** self.t)

                vW_hat = self.v[idx]["vW"] / (1 - self.beta2 ** self.t)
                vB_hat = self.v[idx]["vB"] / (1 - self.beta2 ** self.t)

                # Update weights
                layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.epsilon)
                layer.b -= self.lr * mB_hat / (np.sqrt(vB_hat) + self.epsilon)
                
class RMSProp:
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.v = {}

    def update(self, layers):

        for idx, layer in enumerate(layers):

            if hasattr(layer, "W"):

                if idx not in self.v:
                    self.v[idx] = {
                        "vW": np.zeros_like(layer.W),
                        "vB": np.zeros_like(layer.b)
                    }

                gW = layer.grad_W
                gB = layer.grad_b

                # Update squared gradient moving average
                self.v[idx]["vW"] = self.beta * self.v[idx]["vW"] + (1 - self.beta) * (gW ** 2)
                self.v[idx]["vB"] = self.beta * self.v[idx]["vB"] + (1 - self.beta) * (gB ** 2)

                # Update weights
                layer.W -= self.lr * gW / (np.sqrt(self.v[idx]["vW"]) + self.epsilon)
                layer.b -= self.lr * gB / (np.sqrt(self.v[idx]["vB"]) + self.epsilon)
                
class NAG:
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.lr = learning_rate
        self.beta = beta
        self.velocities = {}

    def update(self, layers):

        for idx, layer in enumerate(layers):

            if hasattr(layer, "W"):

                if idx not in self.velocities:
                    self.velocities[idx] = {
                        "vW": np.zeros_like(layer.W),
                        "vB": np.zeros_like(layer.b)
                    }

                vW_prev = self.velocities[idx]["vW"]
                vB_prev = self.velocities[idx]["vB"]

                # Compute new velocity
                vW = self.beta * vW_prev - self.lr * layer.grad_W
                vB = self.beta * vB_prev - self.lr * layer.grad_b

                # Nesterov update
                layer.W += -self.beta * vW_prev + (1 + self.beta) * vW
                layer.b += -self.beta * vB_prev + (1 + self.beta) * vB

                self.velocities[idx]["vW"] = vW
                self.velocities[idx]["vB"] = vB
                
class Nadam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layers):

        self.t += 1

        for idx, layer in enumerate(layers):

            if hasattr(layer, "W"):

                if idx not in self.m:
                    self.m[idx] = {
                        "mW": np.zeros_like(layer.W),
                        "mB": np.zeros_like(layer.b)
                    }
                    self.v[idx] = {
                        "vW": np.zeros_like(layer.W),
                        "vB": np.zeros_like(layer.b)
                    }

                gW = layer.grad_W
                gB = layer.grad_b

                # Update first moment
                self.m[idx]["mW"] = self.beta1 * self.m[idx]["mW"] + (1 - self.beta1) * gW
                self.m[idx]["mB"] = self.beta1 * self.m[idx]["mB"] + (1 - self.beta1) * gB

                # Update second moment
                self.v[idx]["vW"] = self.beta2 * self.v[idx]["vW"] + (1 - self.beta2) * (gW ** 2)
                self.v[idx]["vB"] = self.beta2 * self.v[idx]["vB"] + (1 - self.beta2) * (gB ** 2)

                # Bias correction
                mW_hat = self.m[idx]["mW"] / (1 - self.beta1 ** self.t)
                mB_hat = self.m[idx]["mB"] / (1 - self.beta1 ** self.t)

                vW_hat = self.v[idx]["vW"] / (1 - self.beta2 ** self.t)
                vB_hat = self.v[idx]["vB"] / (1 - self.beta2 ** self.t)

                # Nadam correction
                mW_nadam = self.beta1 * mW_hat + ((1 - self.beta1) * gW) / (1 - self.beta1 ** self.t)
                mB_nadam = self.beta1 * mB_hat + ((1 - self.beta1) * gB) / (1 - self.beta1 ** self.t)

                # Update weights
                layer.W -= self.lr * mW_nadam / (np.sqrt(vW_hat) + self.epsilon)
                layer.b -= self.lr * mB_nadam / (np.sqrt(vB_hat) + self.epsilon)
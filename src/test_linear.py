import numpy as np
from ann.neural_layer import NeuralLayer  

X = np.random.randn(5, 3)
layer = NeuralLayer(3, 2)

out = layer.forward(X)
print("Forward shape:", out.shape)

dZ = np.random.randn(5, 2)
dX = layer.backward(dZ)

print("grad_W shape:", layer.grad_W.shape)
print("grad_b shape:", layer.grad_b.shape)
print("dX shape:", dX.shape)

from ann.activations import ReLU, Sigmoid, Tanh

x = np.random.randn(4, 3)

relu = ReLU()
print("ReLU:", relu.forward(x))

sig = Sigmoid()
print("Sigmoid:", sig.forward(x))

tanh = Tanh()
print("Tanh:", tanh.forward(x))

from ann.objective_functions import CrossEntropy

# Fake logits
logits = np.random.randn(4, 3)

# Fake one-hot labels
y = np.eye(3)[np.random.randint(0, 3, 4)]

loss_fn = CrossEntropy()
loss = loss_fn.forward(logits, y)
print("Loss:", loss)

dZ = loss_fn.backward()
print("Gradient shape:", dZ.shape)
print("\n"*3)

class Args:
    hidden_size = [64, 32]
    num_layers = 2
    activation = "relu"
    loss = "cross_entropy"
    learning_rate = 0.01

args = Args()

from ann.neural_network import NeuralNetwork

model = NeuralNetwork(args)

X = np.random.randn(4, 784)
y = np.eye(10)[np.random.randint(0, 10, 4)]

logits = model.forward(X)
loss = model.backward(y, logits)

print("Loss:", loss)

print("-"*30)
# Fake small dataset
X = np.random.randn(100, 784)
y = np.eye(10)[np.random.randint(0, 10, 100)]

args.learning_rate = 0.01

model = NeuralNetwork(args)
model.train(X, y, epochs=5, batch_size=16)
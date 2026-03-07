# DA6401 Assignment 1  
## Multi-Layer Perceptron for Image Classification

This repository contains an implementation of a **Multi-Layer Perceptron (MLP)** built completely **from scratch using NumPy** for image classification on the **MNIST** and **Fashion-MNIST** datasets.

The objective of this assignment is to understand the inner workings of neural networks by implementing all major components manually, including forward propagation, backpropagation, loss functions, and optimizers.

---

## Learning Objectives

- Implement **forward propagation and backpropagation**
- Compute gradients manually using NumPy
- Implement different **activation functions**
- Implement multiple **optimization algorithms**
- Train and evaluate neural networks
- Track experiments using **Weights & Biases (W&B)**

---

## Features Implemented

### Neural Network Components
- Fully Connected (Dense) Layers
- Forward Propagation
- Backpropagation

### Activation Functions
- ReLU
- Sigmoid
- Tanh

### Loss Functions
- Cross Entropy Loss
- Mean Squared Error (MSE)

### Optimizers
- SGD
- Momentum
- Nesterov Accelerated Gradient (NAG)
- RMSProp
- Adam
- Nadam

### Weight Initialization
- Random Initialization
- Xavier Initialization

### Experiment Tracking
- Logging metrics using **Weights & Biases**

---

## Dataset

The neural network can be trained on:

- **MNIST** – handwritten digit dataset  
- **Fashion-MNIST** – clothing image dataset  

Datasets are loaded using:
keras.datasets

---

## Project Structure
src/
│
├── ann/
│ ├── neural_network.py # Main neural network implementation
│ ├── layers.py # Fully connected layer implementation
│ ├── activations.py # Activation functions
│ ├── losses.py # Loss functions
│ └── optimizers.py # Optimizer implementations
│
├── train.py # Main training script
│
└── utils.py # Helper functions

Install dependencies:
pip install -r requirements.txt

Training the Model

Example command to train the neural network:

python src/train.py \
--dataset mnist \
--epochs 5 \
--batch_size 32 \
--learning_rate 0.01 \
--optimizer sgd \
--num_layers 2 \
--hidden_size 128 64 \
--activation relu \
--loss cross_entropy \
--weight_init xavier

Weights & Biases Logging

Training metrics such as:
Training Loss
Training Accuracy
Test Accuracy
Gradient Norms
are logged using Weights & Biases (W&B) for experiment tracking.

W&B Report

Complete experiment report for this assignment:
https://wandb.ai/me22b104-iit-madras-foundation/da6401_assignment_1-src/reports/DA6401_report01--VmlldzoxNjEzMzYyNw

GitHub Repo Link: https://github.com/aniket04bit/da6401_assignment_1

Author

Aniket Singh
DA6401 – Deep Learning Assignment 1
IIT Madras

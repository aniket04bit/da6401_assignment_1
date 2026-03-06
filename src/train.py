"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""
print("DEBUG: train.py loaded")

import argparse
import json
import os
import wandb


def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train a neural network')
    
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion_mnist'])

    parser.add_argument('--epochs', type=int, default=5)
    
    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--learning_rate', type=float, default=0.01)

    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'])

    #parser.add_argument('--hidden_size', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--hidden_size', type=int, default=128)
    
    parser.add_argument('--num_layers', type=int, default=2)
    
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'sigmoid', 'tanh'])

    parser.add_argument('--loss', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'mse'])

    parser.add_argument('--weight_init', type=str, default='random')

    parser.add_argument('--wandb_project', type=str, default='dl-assignment')

    parser.add_argument('--model_save_path', type=str, default='models/model.npz')
    
    return parser.parse_args()


"""
The original implementation imported the datasets from
`tensorflow.keras.datasets`.  The project's `requirements.txt` does not
specify `tensorflow` at all – only `keras` – and without a full
TensorFlow installation the import cannot be resolved.  To keep the
dependencies light and to match what's actually installed we simply
import directly from the standalone `keras` package instead.

This also makes the code work with the pip-installed `keras` module,
which provides the same `mnist`/`fashion_mnist` loaders.
"""
from keras.datasets import mnist, fashion_mnist
import numpy as np


def log_sample_images(X, y):
    table = wandb.Table(columns=["image", "label"])

    class_counts = {i: 0 for i in range(10)}

    for i in range(len(X)):
        label = y[i]
        if class_counts[label] < 5:
            image = X[i]
            table.add_data(wandb.Image(image), label)
            class_counts[label] += 1

        if all(count == 5 for count in class_counts.values()):
            break

    wandb.log({"Sample Images per Class": table})
        

def main():

    args = parse_arguments()

    wandb.init(
        project=args.wandb_project,
        config=vars(args)
    )

    config = wandb.config

    # Override args from sweep
    args.learning_rate = config.learning_rate
    args.optimizer = config.optimizer
    args.weight_decay = config.weight_decay
    args.activation = config.activation
    args.batch_size = config.batch_size
    args.hidden_size = config.hidden_size

    # 🔥 Always parse hidden_size safely
    if isinstance(args.hidden_size, str):
        args.hidden_size = [int(x) for x in args.hidden_size.split(",")]

    
    #args.num_layers = len(args.hidden_size)
    
    # Load dataset
    if args.dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    log_sample_images(X_train, y_train)
    
    # Preprocess
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    from ann.neural_network import NeuralNetwork
    model = NeuralNetwork(args)

    # Train
    history = model.train(X_train, y_train, args.epochs, args.batch_size)

    # Log loss
    for epoch, loss in enumerate(history):
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": loss
        })

    # Evaluate
    train_acc = model.evaluate(X_train, y_train)
    test_acc = model.evaluate(X_test, y_test)

    wandb.log({
        "train_accuracy": train_acc,
        "test_accuracy": test_acc
    })

    print(f"Train Accuracy: {train_acc}")
    print(f"Test Accuracy: {test_acc}")

    model.save_model(args.model_save_path)

    wandb.finish()


if __name__ == '__main__':
    main()
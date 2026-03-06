
"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import os
import numpy as np
import wandb
from ann.neural_layer import NeuralLayer
from ann.activations import ReLU, Sigmoid, Tanh
from ann.objective_functions import CrossEntropy, MSE
from ann.optimizers import SGD, Momentum, Adam, RMSProp, NAG, Nadam
class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    def __init__(self, cli_args):

        self.layers = []
        self.weight_decay = cli_args.weight_decay
        input_size = 784
        num_layers = cli_args.num_layers
        hidden_sizes = cli_args.hidden_size

        # ensure hidden_sizes is a list of integers
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes] * num_layers
            
        activation_name = cli_args.activation

        # Choose activation
        if activation_name == "relu":
            activation_class = ReLU
        elif activation_name == "sigmoid":
            activation_class = Sigmoid
        elif activation_name == "tanh":
            activation_class = Tanh
        else:
            raise ValueError("Unsupported activation")

        prev_size = input_size

        # Hidden layers
        for i in range(num_layers):
            self.layers.append(
                NeuralLayer(prev_size, hidden_sizes[i], cli_args.weight_init)
            )
            self.layers.append(activation_class())
            prev_size = hidden_sizes[i]

        # Output layer (10 classes)
        self.layers.append(
            NeuralLayer(prev_size, 10, cli_args.weight_init)
        )

        # Loss
        if cli_args.loss == "cross_entropy":
            self.loss_fn = CrossEntropy()
        else:
            self.loss_fn = MSE()
            
        if cli_args.optimizer == "sgd":
            self.optimizer = SGD(cli_args.learning_rate)

        elif cli_args.optimizer == "momentum":
            self.optimizer = Momentum(cli_args.learning_rate)
            
        elif cli_args.optimizer == "adam":
            self.optimizer = Adam(cli_args.learning_rate)
        
        elif cli_args.optimizer == "rmsprop":
            self.optimizer = RMSProp(cli_args.learning_rate)
            
        elif cli_args.optimizer == "nag":
            self.optimizer = NAG(cli_args.learning_rate)
            
        elif cli_args.optimizer == "nadam":
            self.optimizer = Nadam(cli_args.learning_rate)
    
    def forward(self, X):

        out = X

        for i, layer in enumerate(self.layers):
            out = layer.forward(out)

            # Log hidden layer activations (only during training)
            if hasattr(layer, "__class__"):
                layer_name = layer.__class__.__name__

                # Log only activation layers (ReLU or Tanh)
                if layer_name in ["ReLU", "Tanh"]:
                    zero_fraction = np.mean(out == 0)
                    mean_activation = np.mean(out)

                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log({
                                f"{layer_name}_layer_{i}_zero_fraction": zero_fraction,
                                f"{layer_name}_layer_{i}_mean": mean_activation
                            })
                    except:
                        pass

        return out
    
    def backward(self, y_true, logits):

        # compute loss (needed to store probs and y inside loss_fn)
        self.loss_fn.forward(logits, y_true)

        # gradient of loss w.r.t logits
        dZ = self.loss_fn.backward()

        grad_W_list = []
        grad_b_list = []

        for layer in reversed(self.layers):

            dZ = layer.backward(dZ)

            if hasattr(layer, "W"):
                grad_W_list.append(layer.grad_W.copy())
                grad_b_list.append(layer.grad_b.copy())

        grad_Ws = np.empty(len(grad_W_list), dtype=object)
        grad_bs = np.empty(len(grad_b_list), dtype=object)

        for i in range(len(grad_W_list)):
            grad_Ws[i] = grad_W_list[i]
            grad_bs[i] = grad_b_list[i]

        return grad_Ws, grad_bs
        
    def update_weights(self):
        self.optimizer.update(self.layers)
    
    # def train(self, X_train, y_train, epochs, batch_size):

    #     n = X_train.shape[0]

    #     for epoch in range(epochs):

    #         # Shuffle at start of each epoch
    #         indices = np.random.permutation(n)
    #         X_train = X_train[indices]
    #         y_train = y_train[indices]

    #         epoch_loss = 0.0
    #         num_batches = 0

    #         for i in range(0, n, batch_size):

    #             X_batch = X_train[i:i+batch_size]
    #             y_batch = y_train[i:i+batch_size]

    #             logits = self.forward(X_batch)
    #             loss = self.backward(y_batch, logits)

    #             self.update_weights()

    #             # Accumulate loss
    #             epoch_loss += loss
    #             num_batches += 1

    #         avg_loss = epoch_loss / num_batches

    #         print(f"Epoch {epoch+1}, Avg Loss: {avg_loss}")
    
    def train(self, X_train, y_train, epochs, batch_size):

        n = X_train.shape[0]
        history = []   # store avg loss per epoch

        for epoch in range(epochs):

            # Shuffle at start of each epoch
            indices = np.random.permutation(n)
            X_train = X_train[indices]
            y_train = y_train[indices]

            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, n, batch_size):

                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                logits = self.forward(X_batch)
                loss = self.backward(y_batch, logits)
                
                # Log gradient norm of first hidden layer
                for layer in self.layers:
                    if hasattr(layer, "grad_W"):   # first linear layer
                        grad_norm = np.linalg.norm(layer.grad_W)
                        break

                wandb.log({"grad_norm_layer1": grad_norm})
                
                first_linear = None
                for layer in self.layers:
                    if hasattr(layer, "grad_W"):   # first linear layer
                        first_linear = layer
                        break

                for neuron in range(5):  # log first 5 neurons
                    grad_norm = np.linalg.norm(first_linear.grad_W[:, neuron])
                    wandb.log({f"neuron_{neuron}_grad": grad_norm})
                    
                self.update_weights()

                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            history.append(avg_loss)

            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss}")

        return history

    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """
        logits = self.forward(X)
        preds = np.argmax(logits, axis=1)
        true = np.argmax(y, axis=1)
        accuracy = np.mean(preds == true)
        
        return accuracy


    def save_model(self, path):
        """
        Save model weights to .npz file
        """

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        params = {}
        layer_count = 0

        for layer in self.layers:
            if hasattr(layer, "W"):
                params[f"W{layer_count}"] = layer.W
                params[f"b{layer_count}"] = layer.b
                layer_count += 1

        np.savez(path, **params)
        print(f"Model saved to {path}")
        
    def load_model(self, path):
        data = np.load(path)

        layer_count = 0

        for layer in self.layers:
            if hasattr(layer, "W"):
                layer.W = data[f"W{layer_count}"]
                layer.b = data[f"b{layer_count}"]
                layer_count += 1

        print(f"Model loaded from {path}")
        
    def get_weights(self):
        d = {}
        layer_idx = 0

        for layer in self.layers:
            if hasattr(layer, "W"):
                d[f"W{layer_idx}"] = layer.W.copy()
                d[f"b{layer_idx}"] = layer.b.copy()
                layer_idx += 1

        return d


    def set_weights(self, weight_dict):
        layer_idx = 0

        for layer in self.layers:
            if hasattr(layer, "W"):
                layer.W = weight_dict[f"W{layer_idx}"].copy()
                layer.b = weight_dict[f"b{layer_idx}"].copy()
                layer_idx += 1

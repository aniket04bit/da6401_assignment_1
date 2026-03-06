"""
Inference Script
Evaluate trained models on test sets
"""
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.datasets import mnist, fashion_mnist
import argparse

def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion_mnist'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--hidden_size', type=int, nargs='+', required=True)
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'sigmoid', 'tanh'])

    
    return parser.parse_args()


def load_model(model_path, args):
    """
    Load trained model from disk.
    """

    from ann.neural_network import NeuralNetwork

    # Create dummy args object to rebuild architecture
    class DummyArgs:
        pass

    dummy = DummyArgs()
    dummy.hidden_size = args.hidden_size
    dummy.num_layers = args.num_layers
    dummy.activation = args.activation
    dummy.loss = "cross_entropy"
    dummy.optimizer = "sgd"
    dummy.learning_rate = 0.01
    dummy.weight_init = "random"
    dummy.weight_decay = 0.0

    model = NeuralNetwork(dummy)

    # Load saved weights
    data = np.load(model_path)

    layer_index = 0
    for layer in model.layers:
        if hasattr(layer, "W"):
            layer.W = data[f"W{layer_index}"]
            layer.b = data[f"b{layer_index}"]
            layer_index += 1

    print(f"Model loaded from {model_path}")
    return model


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    logits = model.forward(X_test)

    preds = np.argmax(logits, axis=1)
    true = np.argmax(y_test, axis=1)

    loss = model.loss_fn.forward(logits, y_test)
    accuracy = np.mean(preds == true)

    precision = precision_score(true, preds, average='macro')
    recall = recall_score(true, preds, average='macro')
    f1 = f1_score(true, preds, average='macro')

    return {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def generate_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8,6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    for i in range(10):
        for j in range(10):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()
    plt.show()

    return cm
    
def show_misclassified(X_images, y_true, y_pred, max_images=25):
    import matplotlib.pyplot as plt

    mis_idx = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]

    plt.figure(figsize=(10,10))
    for i in range(min(max_images, len(mis_idx))):
        idx = mis_idx[i]
        plt.subplot(5,5,i+1)
        plt.imshow(X_images[idx], cmap='gray')
        plt.title(f"T:{y_true[idx]} P:{y_pred[idx]}")
        plt.axis('off')

    plt.suptitle("Misclassified Samples")
    plt.tight_layout()
    plt.show()
    
def main():
    """
    Main inference function.
    """
    args = parse_arguments()

    import wandb
    wandb.init(project="dl-assignment", name="confusion-matrix-run")

    # Load dataset
    if args.dataset == 'mnist':
        (_, _), (X_test, y_test_raw) = mnist.load_data()
    else:
        (_, _), (X_test, y_test_raw) = fashion_mnist.load_data()

    # Preprocess
    X_test_flat = X_test.reshape(X_test.shape[0], -1) / 255.0
    y_test = np.eye(10)[y_test_raw]

    # Load model
    model = load_model(args.model_path, args)

    # Forward pass
    logits = model.forward(X_test_flat)
    preds = np.argmax(logits, axis=1)
    true = y_test_raw

    # Evaluate
    results = evaluate_model(model, X_test_flat, y_test)

    print("Loss:", results["loss"])
    print("Accuracy:", results["accuracy"])
    print("Precision:", results["precision"])
    print("Recall:", results["recall"])
    print("F1 Score:", results["f1"])

    logits = model.forward(X_test_flat)
    preds = np.argmax(logits, axis=1)
    true = y_test_raw
    generate_confusion_matrix(true, preds)
    show_misclassified(X_test, true, preds)    
    wandb.finish()

    print("Evaluation complete!")

    return results
    

if __name__ == '__main__':
    main()

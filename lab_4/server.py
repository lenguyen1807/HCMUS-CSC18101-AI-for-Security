import flwr as fl
from sklearn.metrics import accuracy_score
from typing import List, Tuple
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch.nn as nn 
import torch
# Define the Softmax Regression Model
class SoftmaxRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)
    
criterion = nn.CrossEntropyLoss()

test_embeddings = torch.load(f"./embedding_data/test_image_feature.pt")
test_labels = torch.load(f"./embedding_data/test_image_label.pt")
# Number of features and classes
num_features = test_embeddings.shape[1]  # Dimension of the embedding
num_classes = test_labels.max() + 1    # Total classes in the dataset
model = SoftmaxRegression(num_features, num_classes)

# Add noise to model predictions (Output Obfuscation)
def add_noise_to_predictions(predictions, noise_scale=0.1):
    """Add Gaussian noise to predictions for output obfuscation."""
    noise = torch.randn_like(predictions) * noise_scale
    return predictions + noise
import torch

def confidence_thresholding(predictions, threshold=0.7):
    """Apply confidence thresholding to mask low-confidence predictions."""
    probabilities = torch.softmax(predictions, dim=1)
    max_probs, predicted_classes = torch.max(probabilities, dim=1)

    # Mask low-confidence outputs
    mask = max_probs >= threshold
    obfuscated_predictions = torch.zeros_like(probabilities)
    obfuscated_predictions[mask, :] = probabilities[mask, :]

    # Replace masked rows with uniform distribution
    obfuscated_predictions[~mask, :] = 1.0 / predictions.size(1)
    return obfuscated_predictions

# Evaluation function
def evaluate(server_round, parameters, config):
    """Evaluate the global model using the server test set."""
    model = SoftmaxRegression(num_features, num_classes)
    state_dict = zip(model.state_dict().keys(), parameters)
    model.load_state_dict({k: torch.tensor(v) for k, v in state_dict})
    model.eval()
    criterion = nn.CrossEntropyLoss()
    # Test the model on the global test set
    with torch.no_grad():
        outputs = model(test_embeddings)
        # Apply output obfuscation
        # Apply confidence thresholding
        obfuscated_outputs = confidence_thresholding(outputs, threshold=0.7)
        obfuscated_outputs = add_noise_to_predictions(obfuscated_outputs, noise_scale=0.2)
        loss = criterion(obfuscated_outputs, test_labels)
        _, predicted = torch.max(obfuscated_outputs, 1)
        accuracy = (predicted == test_labels).sum().item() / len(test_labels)

    return float(loss.item()), {"accuracy": accuracy}

# Define the strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # All clients participate in each round
    fraction_evaluate=0.0,  # Evaluation done manually on the server
    min_fit_clients=3,
    min_available_clients=3,
    evaluate_fn=evaluate,
)


# Start the Flower server   
if __name__ == "__main__":
    print("Starting Flower server...")
    fl.server.start_server(strategy=strategy,
                           server_address="127.0.0.1:8080", 
                           config=fl.server.ServerConfig(num_rounds=5))

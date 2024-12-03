import flwr as fl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import torch 
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

num_classes = 10
class SoftmaxRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)
    

class MnistClient(fl.client.NumPyClient):
    def __init__(self,  train_embeddings, train_labels, noise_scale=0.1):
        self.train_embeddings = train_embeddings
        self.train_labels = train_labels
        self.num_features = train_embeddings.shape[1]
        self.num_classes = num_classes 
        self.model = SoftmaxRegression (self.num_features, self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)    
        self.noise_scale = noise_scale
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """Update model parameters."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        """Train the model and return updated parameters."""

        self.set_parameters(parameters)
        # Create a DataLoader for the training data
        dataset = TensorDataset(self.train_embeddings, self.train_labels)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        #Train for one epoch
        self.model.train()
        for epoch in range(1):  # Single epoch for simplicity
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

        return self.get_parameters(config), len(self.train_embeddings), {}
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # Create a DataLoader for evaluation data
        dataset = TensorDataset(self.train_embeddings, self.train_labels)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        # Evaluate the model
        self.model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        obfuscated_outputs = []
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                outputs = self.model(X_batch)
                # Add noise for differential privacy
                noise = torch.randn_like(outputs) * self.noise_scale
                outputs += noise
                # Save obfuscated outputs for the server

                loss = self.criterion(outputs, y_batch)
                loss_sum += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)
                
                # Store obfuscated predictions for the attacker
                obfuscated_outputs.append(outputs.cpu().numpy())

        accuracy = correct / total
        return loss_sum / len(dataloader), len(self.train_embeddings), {"accuracy": accuracy, "obfuscated_outputs": obfuscated_outputs}

if __name__ == "__main__":
    import sys

    client_id = int(sys.argv[1])  # Pass client ID as a command-line argument

    print(f"Starting client {client_id}...")
    # ./embedding_data/client1_train_embeddings_.npy
        
    train_embeddings = torch.load(f"./embedding_data/client{client_id}_image_feature.pt")
    train_labels = torch.load(f"./embedding_data/client{client_id}_image_label.pt")
    client = MnistClient(train_embeddings, train_labels)
    fl.client.start_client(
        server_address="127.0.0.1:8080", 
        client=client
    )

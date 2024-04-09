import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm

# Placeholder for DataLoader import
from DataLoader import get_link_prediction_tgb_data

class SpikeNet(nn.Module):
    def __init__(self, in_features, out_features, hids=[32], alpha=1.0, p=0.5, dropout=0.7, bias=True, aggr='mean', sampler='sage', surrogate='triangle', sizes=[5, 2], concat=False, act='LIF'):
        super(SpikeNet, self).__init__()
        self.layers = nn.ModuleList()
        previous_size = in_features
        for hid_size in hids:
            self.layers.append(nn.Linear(previous_size, hid_size))
            previous_size = hid_size
        self.output_layer = nn.Linear(previous_size, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, device, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    return accuracy

def prepare_data_loader(features, labels, batch_size=64, shuffle=True):
    """
    Prepares PyTorch DataLoader from features and labels.
    """
    tensor_x = torch.Tensor(features)  # Convert features to PyTorch tensor
    tensor_y = torch.Tensor(labels).long()  # Convert labels to PyTorch tensor
    dataset = TensorDataset(tensor_x, tensor_y)  # Create tensor dataset
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node_raw_features, _, full_data, train_data, val_data, test_data, _, eval_metric_name = get_link_prediction_tgb_data('tgbn-genre')

    train_loader = prepare_data_loader(train_data.features, train_data.labels, batch_size=64, shuffle=True)
    val_loader = prepare_data_loader(val_data.features, val_data.labels, batch_size=64, shuffle=False)
    test_loader = prepare_data_loader(test_data.features, test_data.labels, batch_size=64, shuffle=False)

    model = SpikeNet(in_features=node_raw_features.shape[1], out_features=np.unique(full_data.labels).size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training and validation loop
    for epoch in range(10):  # Example: 10 epochs
        model.train()
        for data, labels in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation loop
        model.eval()
        val_accuracy = []
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                accuracy = accuracy_score(labels.cpu(), predicted.cpu())
                val_accuracy.append(accuracy)
        
        print(f"Epoch {epoch}: Validation Accuracy: {np.mean(val_accuracy):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "spikenet_tgbl_genre.pth")

    # Evaluate on test data
    model.eval()
    test_accuracy = []
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = accuracy_score(labels.cpu(), predicted.cpu())
            test_accuracy.append(accuracy)
    
    print(f"Test Accuracy: {np.mean(test_accuracy):.4f}")

if __name__ == "__main__":
    main()


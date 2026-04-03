"""
Train a small MLP on the Breast Cancer Wisconsin dataset.
30 input features -> binary classification (malignant/benign).

Uses GELU activation (different from HeartDiseaseNet's SiLU)
to demonstrate the hardened framework generalizes across activations.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import orion.nn as on


class BreastCancerNet(on.Module):
    """
    MLP for breast cancer prediction.
    30 features -> 128 -> 64 -> 2 (class probabilities)

    Uses GELU activation (vs SiLU in HeartDiseaseNet) to exercise
    different polynomial approximation code paths.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = on.Linear(30, 128)
        self.act1 = on.GELU(degree=7)
        self.fc2 = on.Linear(128, 64)
        self.act2 = on.GELU(degree=7)
        self.fc3 = on.Linear(64, 2)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x


def load_data():
    """Load Breast Cancer Wisconsin dataset."""
    print("Loading Breast Cancer Wisconsin dataset...")
    data = load_breast_cancer()
    X, y = data.data.astype(np.float32), data.target.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    return X, y, scaler


def train(model, train_loader, val_X, val_y, epochs=150, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_out = model(val_X)
            val_pred = val_out.argmax(dim=1).numpy()
            val_acc = accuracy_score(val_y.numpy(), val_pred)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 30 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

    model.load_state_dict(best_state)
    return model, best_acc


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    X, y, scaler = load_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y.astype(int))} (malignant / benign)")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_X = torch.tensor(X_train)
    train_y = torch.tensor(y_train)
    val_X = torch.tensor(X_val)
    val_y = torch.tensor(y_val)

    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print("\nTraining BreastCancerNet...")
    model = BreastCancerNet()
    model, best_acc = train(model, train_loader, val_X, val_y, epochs=150)

    model.eval()
    with torch.no_grad():
        val_out = model(val_X)
        val_pred = val_out.argmax(dim=1).numpy()

    print(f"\nBest validation accuracy: {best_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(val_y.numpy(), val_pred, target_names=["Malignant", "Benign"]))

    save_dir = os.path.dirname(__file__)
    model_path = os.path.join(save_dir, "cancer_model.pt")
    scaler_path = os.path.join(save_dir, "cancer_scaler.npz")

    torch.save(model.state_dict(), model_path)
    np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_)

    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

    sample_path = os.path.join(save_dir, "cancer_test_samples.npz")
    np.savez(sample_path, X=X_val, y=y_val)
    print(f"Test samples saved to {sample_path}")


if __name__ == "__main__":
    main()

"""
Train a small MLP on the UCI Heart Disease dataset.
13 input features -> binary classification (heart disease yes/no).

This produces a trained PyTorch model that we'll later run under FHE.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import orion.nn as on


# ─── Heart Disease Model ─────────────────────────────────────────────────────

class HeartDiseaseNet(on.Module):
    """
    Small MLP for heart disease prediction.
    13 features → 64 → 32 → 1 (sigmoid probability)

    Uses orion.nn layers so it can run under FHE with zero code changes.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = on.Linear(13, 64)
        self.act1 = on.SiLU(degree=7)
        self.fc2 = on.Linear(64, 32)
        self.act2 = on.SiLU(degree=7)
        self.fc3 = on.Linear(32, 2)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_heart_disease():
    """Load UCI Heart Disease dataset from OpenML."""
    print("Downloading Heart Disease dataset...")
    heart = fetch_openml(name="heart-statlog", version=1, as_frame=False)
    X, y = heart.data, heart.target

    # Convert target: 1 = disease present, 0 = absent
    # OpenML may return string labels ("present"/"absent") or numeric (1/2)
    if y.dtype.kind in ('U', 'S', 'O'):  # string types
        y = np.array([1.0 if str(v).lower() == "present" else 0.0 for v in y], dtype=np.float32)
    else:
        y = (y.astype(int) - 1).astype(np.float32)

    # Standardize features (important for FHE — keeps values in a manageable range)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    return X, y, scaler


# ─── Training ─────────────────────────────────────────────────────────────────

def train(model, train_loader, val_X, val_y, epochs=100, lr=0.001):
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

        # Validate
        model.eval()
        with torch.no_grad():
            val_out = model(val_X)
            val_pred = val_out.argmax(dim=1).numpy()
            val_acc = accuracy_score(val_y.numpy(), val_pred)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

    model.load_state_dict(best_state)
    return model, best_acc


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    X, y, scaler = load_heart_disease()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y.astype(int))}")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_X = torch.tensor(X_train)
    train_y = torch.tensor(y_train)
    val_X = torch.tensor(X_val)
    val_y = torch.tensor(y_val)

    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Train
    print("\nTraining HeartDiseaseNet...")
    model = HeartDiseaseNet()
    model, best_acc = train(model, train_loader, val_X, val_y, epochs=200)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_out = model(val_X)
        val_pred = val_out.argmax(dim=1).numpy()

    print(f"\nBest validation accuracy: {best_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(val_y.numpy(), val_pred, target_names=["Healthy", "Heart Disease"]))

    # Save model and scaler
    save_dir = os.path.dirname(__file__)
    model_path = os.path.join(save_dir, "heart_model.pt")
    scaler_path = os.path.join(save_dir, "scaler.npz")

    torch.save(model.state_dict(), model_path)
    np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_)

    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

    # Save a test sample for the demo
    sample_path = os.path.join(save_dir, "test_samples.npz")
    np.savez(sample_path, X=X_val, y=y_val)
    print(f"Test samples saved to {sample_path}")


if __name__ == "__main__":
    main()

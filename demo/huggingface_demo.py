"""
HuggingFace Model → FHE Inference Demo.

Demonstrates how to take a standard PyTorch model (as you'd find on
HuggingFace Hub) and run it under Fully Homomorphic Encryption.

Three examples:
  1. Convert a torch.nn model to Orion and run FHE inference
  2. Check compatibility of various model architectures
  3. Show what happens with unsupported models (Transformers)

Usage:
    python demo/huggingface_demo.py
"""

import os
import sys
import time

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orion.integrations import convert_to_orion, check_compatibility


# ================================================================
#  Example 1: Standard PyTorch MLP → FHE
# ================================================================

def example_convert_and_infer():
    """
    Train a standard torch.nn MLP on Iris, convert to Orion, run FHE.

    This simulates the workflow: download model from HuggingFace →
    check compatibility → convert → encrypt → infer → decrypt.
    """
    print("=" * 60)
    print("Example 1: Standard PyTorch MLP → FHE Inference")
    print("=" * 60)

    # ----- Step 1: Train a standard PyTorch model -----
    print("\n[1] Training standard PyTorch model on Iris dataset...")

    torch.manual_seed(42)
    np.random.seed(42)

    data = load_iris()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # This is a STANDARD torch.nn model — no Orion imports needed!
    class IrisClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 32)
            self.act1 = nn.SiLU()
            self.fc2 = nn.Linear(32, 16)
            self.act2 = nn.SiLU()
            self.fc3 = nn.Linear(16, 3)

        def forward(self, x):
            x = self.act1(self.fc1(x))
            x = self.act2(self.fc2(x))
            return self.fc3(x)

    model = IrisClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    train_X = torch.tensor(X_train)
    train_y = torch.tensor(y_train)

    for epoch in range(200):
        optimizer.zero_grad()
        out = model(train_X)
        loss = criterion(out, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_X = torch.tensor(X_test)
        preds = model(test_X).argmax(dim=1).numpy()
        acc = (preds == y_test).mean()
    print(f"    Cleartext accuracy: {acc:.1%}")

    # ----- Step 2: Check FHE compatibility -----
    print("\n[2] Checking FHE compatibility...")
    report = check_compatibility(model, "IrisClassifier")
    print(report)

    if not report.compatible:
        print("Model not compatible!")
        return

    # ----- Step 3: Convert to Orion -----
    print("\n[3] Converting to Orion FHE model...")
    orion_model = convert_to_orion(model, activation_degree=7)
    print(f"    Converted! Type: {type(orion_model).__name__}")

    # Verify cleartext still works after conversion
    orion_model.eval()
    with torch.no_grad():
        orion_preds = orion_model(test_X).argmax(dim=1).numpy()
        orion_acc = (orion_preds == y_test).mean()
    print(f"    Post-conversion cleartext accuracy: {orion_acc:.1%}")
    print(f"    Weight transfer verified: {np.array_equal(preds, orion_preds)}")

    # ----- Step 4: Run FHE inference -----
    print("\n[4] Running FHE inference...")
    import orion

    config_path = os.path.join(os.path.dirname(__file__), "heart_config.yml")
    if not os.path.exists(config_path):
        print("    Skipping FHE (heart_config.yml not found). "
              "Run train_model.py first.")
        return

    scheme = orion.init_scheme(config_path)

    fit_dataset = TensorDataset(torch.tensor(X_train), torch.zeros(len(X_train)))
    orion.fit(orion_model, DataLoader(fit_dataset, batch_size=32))

    input_level = orion.compile(orion_model)
    print(f"    Input level: {input_level}")

    # Test a few samples
    num_test = 5
    orion_model.he()
    correct = 0
    labels = data.target_names

    for i in range(num_test):
        sample = torch.tensor(X_test[i:i+1], dtype=torch.float32)
        actual = int(y_test[i])

        t0 = time.time()
        ptxt = orion.encode(sample, input_level)
        ctxt = orion.encrypt(ptxt)
        out_ctxt = orion_model(ctxt)
        out_fhe = out_ctxt.decrypt().decode()
        t_total = time.time() - t0

        pred = out_fhe.flatten()[:3].argmax().item()
        if pred == actual:
            correct += 1
        status = "ok" if pred == actual else "WRONG"

        print(f"    Sample {i+1}/{num_test}: {labels[pred]:>12s} "
              f"(actual: {labels[actual]:>12s}) [{status}] | {t_total:.2f}s")

    print(f"\n    FHE Accuracy: {correct}/{num_test}")
    scheme.delete_scheme()


# ================================================================
#  Example 2: Compatibility Checks
# ================================================================

def example_compatibility_checks():
    """Show compatibility reports for various architectures."""
    print("\n" + "=" * 60)
    print("Example 2: Compatibility Checks for Various Architectures")
    print("=" * 60)

    # 2a: Simple MLP — COMPATIBLE
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(784, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 10),
            )
        def forward(self, x):
            return self.layers(x)

    print("\n--- Simple MLP (MNIST-style) ---")
    report = check_compatibility(SimpleMLP(), "SimpleMLP")
    print(report)

    # 2b: CNN — COMPATIBLE
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.SiLU(),
                nn.AvgPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32, 10),
            )
        def forward(self, x):
            return self.classifier(self.features(x))

    print("\n--- Simple CNN (CIFAR-style) ---")
    report = check_compatibility(SimpleCNN(), "SimpleCNN")
    print(report)

    # 2c: Model with MaxPool — NOT COMPATIBLE
    class CNNWithMaxPool(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3)
            self.pool = nn.MaxPool2d(2)  # BLOCKER
            self.fc = nn.Linear(16 * 13 * 13, 10)
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            return self.fc(x.flatten(1))

    print("\n--- CNN with MaxPool (NOT compatible) ---")
    report = check_compatibility(CNNWithMaxPool(), "CNNWithMaxPool")
    print(report)

    # 2d: Transformer — NOT COMPATIBLE
    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 64)
            encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.fc = nn.Linear(64, 10)
        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer(x)
            return self.fc(x.mean(dim=1))

    print("\n--- Transformer (NOT compatible) ---")
    report = check_compatibility(TinyTransformer(), "TinyTransformer")
    print(report)


# ================================================================
#  Example 3: Conversion with various activations
# ================================================================

def example_activation_variety():
    """Show that different activations all convert correctly."""
    print("\n" + "=" * 60)
    print("Example 3: Various Activation Functions")
    print("=" * 60)

    activations = [nn.ReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.ELU,
                   nn.SELU, nn.Mish, nn.Softplus]

    for act_class in activations:
        model = nn.Sequential(
            nn.Linear(10, 32),
            act_class(),
            nn.Linear(32, 2),
        )
        # Random weights
        nn.init.xavier_uniform_(model[0].weight)
        nn.init.xavier_uniform_(model[2].weight)

        orion_model = convert_to_orion(model, activation_degree=7)

        # Verify cleartext equivalence
        test_input = torch.randn(5, 10)
        model.eval()
        orion_model.eval()
        with torch.no_grad():
            orig_out = model(test_input)
            conv_out = orion_model(test_input)
            diff = (orig_out - conv_out).abs().max().item()

        status = "MATCH" if diff < 1e-5 else f"diff={diff:.6f}"
        print(f"  {act_class.__name__:>10s}: {status}")


def main():
    example_compatibility_checks()
    example_activation_variety()
    example_convert_and_infer()


if __name__ == "__main__":
    main()

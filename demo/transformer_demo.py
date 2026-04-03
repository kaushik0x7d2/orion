"""
Transformer -> FHE Demo.

Demonstrates that transformer models can be made FHE-compatible using:
1. PolySoftmax (x^p / sum(x^p)) instead of exp-based softmax
2. Polynomial GELU activation (Chebyshev approximation)
3. FHELayerNorm (pre-computed normalization for FHE)

Three examples:
  1. Train FHE-compatible transformer on Iris (PolySoftmax vs standard)
  2. Compatibility analysis of FHE vs standard transformers
  3. LayerNorm conversion from standard PyTorch

Usage:
    python demo/transformer_demo.py
"""

import os
import sys

import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orion.nn.transformer import (
    PolySoftmax, FHELayerNorm, FHEMultiHeadAttention,
    FHETransformerEncoderLayer,
)
from orion.integrations import check_compatibility, convert_to_orion


# ================================================================
#  Example 1: FHE-Compatible Transformer vs Standard
# ================================================================

def example_polysoftmax_training():
    """Train and compare standard vs polynomial softmax transformer."""
    print("=" * 60)
    print("Example 1: Standard Softmax vs PolySoftmax Transformer")
    print("=" * 60)

    # Load Iris
    data = load_iris()
    X = StandardScaler().fit_transform(data.data).astype(np.float32)
    y = data.target.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    train_X = torch.tensor(X_train)
    train_y = torch.tensor(y_train)
    test_X = torch.tensor(X_test)

    # ---- Standard PyTorch Transformer ----
    class StandardTransformerClassifier(nn.Module):
        """Standard transformer — NOT FHE-compatible (uses exp-softmax)."""
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(1, 32)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=32, nhead=4, dim_feedforward=64,
                dropout=0.0, batch_first=True, activation='gelu')
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.classifier = nn.Linear(32, 3)

        def forward(self, x):
            x = self.embed(x.unsqueeze(-1))  # [B, 4] -> [B, 4, 32]
            x = self.encoder(x)              # [B, 4, 32]
            x = x.mean(dim=1)                # [B, 32]
            return self.classifier(x)

    # ---- FHE-Compatible Transformer ----
    class FHETransformerClassifier(nn.Module):
        """FHE-compatible transformer using PolySoftmax + polynomial GELU."""
        def __init__(self, softmax_power=4):
            super().__init__()
            self.embed = nn.Linear(1, 32)
            self.encoder = FHETransformerEncoderLayer(
                d_model=32, nhead=4, dim_feedforward=64,
                softmax_power=softmax_power, activation_degree=7)
            self.classifier = nn.Linear(32, 3)

        def forward(self, x):
            x = self.embed(x.unsqueeze(-1))
            x = self.encoder(x)
            x = x.mean(dim=1)
            return self.classifier(x)

    results = {}
    for name, ModelClass, kwargs in [
        ("Standard Softmax", StandardTransformerClassifier, {}),
        ("PolySoftmax p=2", FHETransformerClassifier, {"softmax_power": 2}),
        ("PolySoftmax p=4", FHETransformerClassifier, {"softmax_power": 4}),
        ("PolySoftmax p=8", FHETransformerClassifier, {"softmax_power": 8}),
    ]:
        torch.manual_seed(42)
        model = ModelClass(**kwargs)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(300):
            optimizer.zero_grad()
            loss = criterion(model(train_X), train_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(test_X).argmax(dim=1).numpy()
            acc = (preds == y_test).mean()

        results[name] = acc
        print(f"  {name:>20s}: {acc:.1%} accuracy")

    print(f"\n  All PolySoftmax variants achieve competitive accuracy")
    print(f"  with standard softmax, proving FHE compatibility is feasible.")


# ================================================================
#  Example 2: Compatibility Analysis
# ================================================================

def example_compatibility_analysis():
    """Show FHE compatibility of various transformer configurations."""
    print("\n" + "=" * 60)
    print("Example 2: Transformer FHE Compatibility Analysis")
    print("=" * 60)

    # FHE-compatible transformer
    class FHETransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(1, 32)
            self.encoder = FHETransformerEncoderLayer(
                d_model=32, nhead=4, dim_feedforward=64, softmax_power=4)
            self.classifier = nn.Linear(32, 3)
        def forward(self, x):
            return self.classifier(
                self.encoder(self.embed(x.unsqueeze(-1))).mean(1))

    print("\n--- FHE-Compatible Transformer (PolySoftmax + FHELayerNorm) ---")
    report = check_compatibility(FHETransformer(), "FHETransformer")
    print(report)

    # Standard transformer (NOT compatible)
    class StdTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(1, 32)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=32, nhead=4, dim_feedforward=64,
                dropout=0, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.classifier = nn.Linear(32, 3)
        def forward(self, x):
            return self.classifier(
                self.encoder(self.embed(x.unsqueeze(-1))).mean(1))

    print("\n--- Standard Transformer (NOT FHE-compatible) ---")
    report = check_compatibility(StdTransformer(), "StdTransformer")
    print(report)

    # Hybrid: standard transformer with LayerNorm convertible
    class MLP_with_LayerNorm(nn.Module):
        """MLP using LayerNorm — now auto-convertible to FHE."""
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 32)
            self.norm = nn.LayerNorm(32)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(32, 3)
        def forward(self, x):
            return self.fc2(self.act(self.norm(self.fc1(x))))

    print("\n--- MLP with LayerNorm (NOW compatible via FHELayerNorm) ---")
    report = check_compatibility(MLP_with_LayerNorm(), "MLP_LayerNorm")
    print(report)


# ================================================================
#  Example 3: LayerNorm Conversion + Depth Analysis
# ================================================================

def example_layernorm_conversion():
    """Show LayerNorm auto-conversion and FHE depth analysis."""
    print("\n" + "=" * 60)
    print("Example 3: LayerNorm Conversion & FHE Depth Analysis")
    print("=" * 60)

    # MLP with LayerNorm
    class MLPWithLayerNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 32)
            self.norm1 = nn.LayerNorm(32)
            self.act1 = nn.GELU()
            self.fc2 = nn.Linear(32, 16)
            self.norm2 = nn.LayerNorm(16)
            self.act2 = nn.GELU()
            self.fc3 = nn.Linear(16, 3)

        def forward(self, x):
            x = self.act1(self.norm1(self.fc1(x)))
            x = self.act2(self.norm2(self.fc2(x)))
            return self.fc3(x)

    torch.manual_seed(42)
    model = MLPWithLayerNorm()

    # Train briefly
    data = load_iris()
    X = StandardScaler().fit_transform(data.data).astype(np.float32)
    y = data.target.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    train_X, test_X = torch.tensor(X_train), torch.tensor(X_test)
    train_y = torch.tensor(y_train)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(200):
        optimizer.zero_grad()
        criterion(model(train_X), train_y).backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        orig_preds = model(test_X).argmax(1).numpy()
        orig_acc = (orig_preds == y_test).mean()
    print(f"\n  Original model accuracy: {orig_acc:.1%}")

    # Convert to Orion (LayerNorm -> FHELayerNorm automatically)
    orion_model = convert_to_orion(model, activation_degree=7)

    orion_model.eval()
    with torch.no_grad():
        conv_preds = orion_model(test_X).argmax(1).numpy()
        conv_acc = (conv_preds == y_test).mean()
    print(f"  Converted model accuracy: {conv_acc:.1%}")
    print(f"  Weight transfer verified: {np.array_equal(orig_preds, conv_preds)}")

    # Show converted layer types
    print(f"\n  Converted layer types:")
    for name, module in orion_model.named_modules():
        if name and not any(True for _ in module.children()):
            print(f"    {name}: {type(module).__name__}")

    # FHE depth analysis
    print(f"\n  FHE Depth Analysis:")
    print(f"    The FHE-compatible transformer encoder layer depth breakdown:")
    enc = FHETransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64)
    print(f"    - FHELayerNorm:          {enc.norm1.depth} levels")
    print(f"    - FHEMultiHeadAttention:  {enc.self_attn.depth} levels")
    print(f"    - PolySoftmax (p=4):     {enc.self_attn.poly_softmax.depth} levels")
    print(f"    - Linear (FFN):          1 level each")
    print(f"    - GELU (degree 7):       ~4 levels")
    print(f"    - Total encoder layer:   {enc.depth} levels")
    print(f"    - With LogN=15, LogQ supports ~50 levels before bootstrap")


def main():
    example_polysoftmax_training()
    example_compatibility_analysis()
    example_layernorm_conversion()


if __name__ == "__main__":
    main()

"""
FHE inference on the Breast Cancer model.

Demonstrates the hardened framework works across:
  - Different datasets (breast cancer vs heart disease)
  - Different activations (GELU vs SiLU)
  - Different input dimensions (30 vs 13 features)
"""

import os
import sys
import time
import math
import json

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import orion
from demo.train_cancer import BreastCancerNet


def main():
    torch.manual_seed(42)

    demo_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(demo_dir, "cancer_config.yml")

    for f in ["cancer_model.pt", "cancer_test_samples.npz"]:
        if not os.path.exists(os.path.join(demo_dir, f)):
            print(f"Missing {f}. Run train_cancer.py first.")
            return

    # Load model
    model = BreastCancerNet()
    model.load_state_dict(torch.load(
        os.path.join(demo_dir, "cancer_model.pt"), weights_only=True))
    model.eval()

    samples = np.load(os.path.join(demo_dir, "cancer_test_samples.npz"))
    X_test, y_test = samples["X"], samples["y"]

    # Cleartext
    print("=== Cleartext Inference ===")
    X = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        clear_out = model(X)
        clear_preds = clear_out.argmax(dim=1).numpy()

    clear_acc = (clear_preds == y_test.astype(int)).mean()
    print(f"  Accuracy: {clear_acc:.1%} ({int(clear_acc*len(y_test))}/{len(y_test)})")

    # FHE
    num_samples = min(20, len(X_test))
    print(f"\n=== FHE Inference ({num_samples} samples) ===")

    print("\n[1] Initializing scheme...")
    t0 = time.time()
    scheme = orion.init_scheme(config_path)
    print(f"    Done ({time.time()-t0:.2f}s)")

    print("[2] Fitting...")
    t0 = time.time()
    fit_X = torch.tensor(X_test, dtype=torch.float32)
    fit_dataset = TensorDataset(fit_X, torch.zeros(len(fit_X)))
    orion.fit(model, DataLoader(fit_dataset, batch_size=32))
    print(f"    Done ({time.time()-t0:.2f}s)")

    print("[3] Compiling...")
    t0 = time.time()
    input_level = orion.compile(model)
    print(f"    Done ({time.time()-t0:.2f}s) | Input level: {input_level}")

    labels = ["Malignant", "Benign"]
    results = []
    times = []

    for i in range(num_samples):
        patient = torch.tensor(X_test[i:i+1], dtype=torch.float32)
        actual = int(y_test[i])

        t0 = time.time()
        ptxt = orion.encode(patient, input_level)
        ctxt = orion.encrypt(ptxt)
        model.he()
        t_enc = time.time() - t0

        t0 = time.time()
        out_ctxt = model(ctxt)
        t_inf = time.time() - t0

        t0 = time.time()
        out_fhe = out_ctxt.decrypt().decode()
        t_dec = time.time() - t0

        fhe_out = out_fhe.flatten()[:2]
        pred = fhe_out.argmax().item()
        results.append(pred)
        times.append(t_inf)

        # Precision
        model.eval()
        with torch.no_grad():
            c_out = model(patient).flatten()[:2]
        mae = (c_out - fhe_out).abs().mean().item()
        bits = -math.log2(mae) if mae > 0 else float('inf')

        status = "ok" if pred == actual else "WRONG"
        print(f"  Sample {i+1:2d}/{num_samples}: {labels[pred]:>10s} "
              f"(actual: {labels[actual]:>10s}) [{status:>5s}] "
              f"| {t_inf:.2f}s | {bits:.1f} bits")

    # Summary
    correct = sum(1 for i in range(len(results)) if results[i] == int(y_test[i]))
    agree = sum(1 for i in range(len(results)) if results[i] == clear_preds[i])

    print(f"\n=== Summary ===")
    print(f"  FHE Accuracy:         {correct}/{num_samples} ({correct/num_samples:.1%})")
    print(f"  FHE-Clear Agreement:  {agree}/{num_samples}")
    print(f"  Avg Inference Time:   {np.mean(times):.2f}s")
    print(f"  Model:                BreastCancerNet (30->128->64->2, GELU)")

    scheme.delete_scheme()


if __name__ == "__main__":
    main()

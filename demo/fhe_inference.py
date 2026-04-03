"""
End-to-end FHE inference on the Heart Disease model.

This script demonstrates the full pipeline:
  1. Load trained model
  2. Initialize FHE scheme
  3. Fit & compile the model for FHE
  4. Encrypt patient data
  5. Run inference on encrypted data
  6. Decrypt and compare results

The server never sees the patient's raw data or the prediction.
"""

import os
import sys
import time
import math

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import orion
from demo.train_model import HeartDiseaseNet


def load_model_and_data(demo_dir):
    """Load the trained model, scaler, and test samples."""
    model = HeartDiseaseNet()
    model.load_state_dict(torch.load(
        os.path.join(demo_dir, "heart_model.pt"), weights_only=True))
    model.eval()

    scaler = np.load(os.path.join(demo_dir, "scaler.npz"))
    samples = np.load(os.path.join(demo_dir, "test_samples.npz"))

    return model, scaler, samples["X"], samples["y"]


def run_cleartext(model, X_test, y_test, num_samples=5):
    """Run cleartext inference for comparison."""
    print("=== Cleartext Inference ===")
    X = torch.tensor(X_test[:num_samples], dtype=torch.float32)
    y = y_test[:num_samples]

    with torch.no_grad():
        out = model(X)
        preds = out.argmax(dim=1).numpy()

    labels = ["Healthy", "Heart Disease"]
    for i in range(num_samples):
        status = "correct" if preds[i] == int(y[i]) else "WRONG"
        print(f"  Patient {i+1}: {labels[int(preds[i])]:>14s}  (actual: {labels[int(y[i])]:>14s}) [{status}]")

    acc = (preds == y.astype(int)).mean()
    print(f"  Accuracy: {acc:.0%}")
    return out, preds


def run_fhe_inference(model, X_test, y_test, config_path, num_samples=5):
    """Run FHE inference — the core demo."""
    print("\n=== FHE (Encrypted) Inference ===")

    # Step 1: Init FHE scheme
    print("\n[1/6] Initializing FHE scheme...")
    t0 = time.time()
    scheme = orion.init_scheme(config_path)
    print(f"      Done ({time.time()-t0:.2f}s)")

    # Step 2: Fit model (collect value ranges for polynomial approximations)
    print("\n[2/6] Fitting model for FHE...")
    t0 = time.time()
    # Use all test data for range estimation
    fit_X = torch.tensor(X_test, dtype=torch.float32)
    fit_dataset = TensorDataset(fit_X, torch.zeros(len(fit_X)))
    fit_loader = DataLoader(fit_dataset, batch_size=32)
    orion.fit(model, fit_loader)
    print(f"      Done ({time.time()-t0:.2f}s)")

    # Step 3: Compile (pack weights, find bootstrap locations)
    print("\n[3/6] Compiling model for FHE...")
    t0 = time.time()
    input_level = orion.compile(model)
    print(f"      Done ({time.time()-t0:.2f}s) | Input level: {input_level}")

    # Now run inference on each sample
    results = []
    labels = ["Healthy", "Heart Disease"]

    for i in range(num_samples):
        patient_data = torch.tensor(X_test[i:i+1], dtype=torch.float32)
        actual = int(y_test[i])

        # Step 4: Encrypt
        print(f"\n[4/6] Encrypting patient {i+1} data...")
        t0 = time.time()
        ptxt = orion.encode(patient_data, input_level)
        ctxt = orion.encrypt(ptxt)
        model.he()
        t_enc = time.time() - t0

        # Step 5: FHE Inference (this is where the magic happens)
        print(f"[5/6] Running encrypted inference...")
        t0 = time.time()
        out_ctxt = model(ctxt)
        t_inf = time.time() - t0

        # Step 6: Decrypt
        print(f"[6/6] Decrypting result...")
        t0 = time.time()
        out_ptxt = out_ctxt.decrypt()
        out_fhe = out_ptxt.decode()
        t_dec = time.time() - t0

        pred = out_fhe[:2].argmax().item()
        results.append(pred)

        status = "correct" if pred == actual else "WRONG"
        print(f"      Patient {i+1}: {labels[pred]:>14s}  (actual: {labels[actual]:>14s}) [{status}]")
        print(f"      Encrypt: {t_enc:.3f}s | Inference: {t_inf:.3f}s | Decrypt: {t_dec:.3f}s | Total: {t_enc+t_inf+t_dec:.3f}s")

    # Compare with cleartext
    print("\n=== Summary ===")
    fhe_acc = sum(1 for i in range(num_samples) if results[i] == int(y_test[i])) / num_samples
    print(f"FHE Accuracy: {fhe_acc:.0%} ({sum(1 for i in range(num_samples) if results[i] == int(y_test[i]))}/{num_samples})")

    # Compute precision vs cleartext
    model.eval()  # switch back to cleartext mode
    with torch.no_grad():
        clear_out = model(torch.tensor(X_test[:1], dtype=torch.float32))

    # Re-run one sample in FHE mode for precision measurement
    ptxt = orion.encode(torch.tensor(X_test[:1], dtype=torch.float32), input_level)
    ctxt = orion.encrypt(ptxt)
    model.he()
    fhe_out = model(ctxt).decrypt().decode()[:2]
    clear_flat = clear_out.detach().flatten()[:2]

    mae = (clear_flat - fhe_out).abs().mean().item()
    if mae > 0:
        precision = -math.log2(mae)
        print(f"Precision vs cleartext: {precision:.1f} bits (MAE={mae:.6f})")
    else:
        print("Precision vs cleartext: exact match")

    scheme.delete_scheme()
    return results


def main():
    torch.manual_seed(42)

    demo_dir = os.path.dirname(__file__)
    config_path = os.path.join(demo_dir, "heart_config.yml")

    # Check files exist
    for f in ["heart_model.pt", "scaler.npz", "test_samples.npz"]:
        if not os.path.exists(os.path.join(demo_dir, f)):
            print(f"Missing {f}. Run train_model.py first.")
            return

    model, scaler, X_test, y_test = load_model_and_data(demo_dir)

    num_samples = 5
    clear_out, clear_preds = run_cleartext(model, X_test, y_test, num_samples)
    fhe_results = run_fhe_inference(model, X_test, y_test, config_path, num_samples)

    # Final comparison
    print("\n=== Cleartext vs FHE Agreement ===")
    agree = sum(1 for i in range(num_samples) if clear_preds[i] == fhe_results[i])
    print(f"Predictions agree: {agree}/{num_samples}")


if __name__ == "__main__":
    main()

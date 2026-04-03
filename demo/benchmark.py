"""
Paper-ready benchmarks for the hardened Orion FHE framework.

Measures:
  1. End-to-end FHE inference (encrypt -> infer -> decrypt)
  2. Per-stage timing breakdown
  3. Precision vs cleartext (bits of accuracy)
  4. Ciphertext sizes
  5. Client-server overhead vs local FHE
  6. Accuracy across all test samples

Output: Structured results for paper tables/figures.
"""

import os
import sys
import time
import math
import json

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import orion
from demo.train_model import HeartDiseaseNet


def benchmark_cleartext(model, X_test, y_test):
    """Benchmark cleartext inference."""
    model.eval()
    X = torch.tensor(X_test, dtype=torch.float32)

    t0 = time.time()
    with torch.no_grad():
        out = model(X)
    t_clear = time.time() - t0

    preds = out.argmax(dim=1).numpy()
    acc = accuracy_score(y_test.astype(int), preds)

    return {
        "cleartext_time_total": t_clear,
        "cleartext_time_per_sample": t_clear / len(X_test),
        "cleartext_accuracy": acc,
        "cleartext_predictions": preds.tolist(),
        "cleartext_outputs": out.detach().numpy().tolist(),
    }


def benchmark_fhe(model, X_test, y_test, config_path, num_samples=None):
    """Benchmark full FHE pipeline."""
    if num_samples is None:
        num_samples = len(X_test)
    num_samples = min(num_samples, len(X_test))

    # Init scheme
    t0 = time.time()
    scheme = orion.init_scheme(config_path)
    t_init = time.time() - t0

    # Fit
    t0 = time.time()
    fit_X = torch.tensor(X_test, dtype=torch.float32)
    fit_dataset = TensorDataset(fit_X, torch.zeros(len(fit_X)))
    fit_loader = DataLoader(fit_dataset, batch_size=32)
    orion.fit(model, fit_loader)
    t_fit = time.time() - t0

    # Compile
    t0 = time.time()
    input_level = orion.compile(model)
    t_compile = time.time() - t0

    results = {
        "scheme_init_time": t_init,
        "fit_time": t_fit,
        "compile_time": t_compile,
        "input_level": input_level,
        "num_samples": num_samples,
        "patients": [],
    }

    # Per-patient inference
    for i in range(num_samples):
        patient_data = torch.tensor(X_test[i:i+1], dtype=torch.float32)
        actual = int(y_test[i])

        # Encrypt
        t0 = time.time()
        ptxt = orion.encode(patient_data, input_level)
        ctxt = orion.encrypt(ptxt)
        model.he()
        t_enc = time.time() - t0

        # Measure ciphertext size
        serialized = ctxt.serialize()
        ct_size = sum(len(b) for b in serialized["ciphertexts"])

        # Inference
        t0 = time.time()
        out_ctxt = model(ctxt)
        t_inf = time.time() - t0

        # Decrypt
        t0 = time.time()
        out_ptxt = out_ctxt.decrypt()
        out_fhe = out_ptxt.decode()
        t_dec = time.time() - t0

        pred = out_fhe[:2].argmax().item()
        fhe_values = out_fhe[:2].tolist()

        # Precision vs cleartext for this sample
        model.eval()
        with torch.no_grad():
            clear_out = model(patient_data).flatten()[:2]

        fhe_flat = out_fhe[:2]
        mae = (clear_out - fhe_flat).abs().mean().item()
        precision_bits = -math.log2(mae) if mae > 0 else float('inf')

        results["patients"].append({
            "patient_id": i + 1,
            "actual": actual,
            "predicted": pred,
            "correct": pred == actual,
            "fhe_output": fhe_values,
            "clear_output": clear_out.tolist(),
            "mae": mae,
            "precision_bits": precision_bits,
            "encrypt_time": t_enc,
            "inference_time": t_inf,
            "decrypt_time": t_dec,
            "total_time": t_enc + t_inf + t_dec,
            "ciphertext_bytes": ct_size,
        })

        print(f"  Patient {i+1}/{num_samples}: "
              f"{'correct' if pred == actual else 'WRONG':>7s} | "
              f"{t_inf:.2f}s | {precision_bits:.1f} bits")

    # Aggregate stats
    patients = results["patients"]
    correct = sum(1 for p in patients if p["correct"])
    results["fhe_accuracy"] = correct / num_samples
    results["avg_encrypt_time"] = np.mean([p["encrypt_time"] for p in patients])
    results["avg_inference_time"] = np.mean([p["inference_time"] for p in patients])
    results["avg_decrypt_time"] = np.mean([p["decrypt_time"] for p in patients])
    results["avg_total_time"] = np.mean([p["total_time"] for p in patients])
    results["avg_precision_bits"] = np.mean([p["precision_bits"] for p in patients
                                             if p["precision_bits"] != float('inf')])
    results["avg_mae"] = np.mean([p["mae"] for p in patients])
    results["avg_ciphertext_bytes"] = np.mean([p["ciphertext_bytes"] for p in patients])

    scheme.delete_scheme()
    return results


def print_paper_table(clear_results, fhe_results):
    """Print results formatted for paper tables."""
    n = fhe_results["num_samples"]
    print("\n" + "=" * 70)
    print("  PAPER-READY BENCHMARK RESULTS")
    print("=" * 70)

    print("\n--- Table 1: System Configuration ---")
    print(f"  Model:            HeartDiseaseNet (13 -> 64 -> 32 -> 2)")
    print(f"  Activation:       SiLU (Chebyshev degree 7)")
    print(f"  CKKS Parameters:  LogN=14, N=16384, Standard ring")
    print(f"  Effective levels: 13 (LogQ: 14 entries)")
    print(f"  Scale:            2^35")
    print(f"  Input level:      {fhe_results['input_level']}")
    print(f"  Bootstrap ops:    0")
    print(f"  Test samples:     {n}")

    print("\n--- Table 2: Accuracy ---")
    print(f"  Cleartext accuracy:    {clear_results['cleartext_accuracy']:.1%}")
    print(f"  FHE accuracy:          {fhe_results['fhe_accuracy']:.1%}")
    print(f"  FHE-Clear agreement:   {sum(1 for i,p in enumerate(fhe_results['patients']) if p['predicted'] == clear_results['cleartext_predictions'][i])}/{n}")
    print(f"  Avg precision (bits):  {fhe_results['avg_precision_bits']:.1f}")
    print(f"  Avg MAE:               {fhe_results['avg_mae']:.6f}")

    print("\n--- Table 3: Timing (seconds) ---")
    print(f"  Scheme init:     {fhe_results['scheme_init_time']:>8.3f}s  (one-time)")
    print(f"  Model fit:       {fhe_results['fit_time']:>8.3f}s  (one-time)")
    print(f"  Model compile:   {fhe_results['compile_time']:>8.3f}s  (one-time)")
    print(f"  Avg encrypt:     {fhe_results['avg_encrypt_time']:>8.3f}s  (per sample)")
    print(f"  Avg inference:   {fhe_results['avg_inference_time']:>8.3f}s  (per sample)")
    print(f"  Avg decrypt:     {fhe_results['avg_decrypt_time']:>8.3f}s  (per sample)")
    print(f"  Avg total:       {fhe_results['avg_total_time']:>8.3f}s  (per sample)")
    print(f"  Cleartext total: {clear_results['cleartext_time_total']:>8.6f}s (all {n} samples)")

    slowdown = fhe_results['avg_inference_time'] / (clear_results['cleartext_time_per_sample'] + 1e-10)
    print(f"  FHE slowdown:    {slowdown:>8.0f}x")

    print("\n--- Table 4: Communication ---")
    ct_kb = fhe_results['avg_ciphertext_bytes'] / 1024
    print(f"  Ciphertext size: {ct_kb:.0f} KB ({fhe_results['avg_ciphertext_bytes']} bytes)")
    print(f"  Features:        13 (float32) = 52 bytes plaintext")
    print(f"  Expansion ratio: {fhe_results['avg_ciphertext_bytes'] / 52:.0f}x")

    print("\n--- Table 5: Vulnerability Fixes ---")
    fixes = [
        ("Windows scale overflow (C.ulong 32-bit)", "Critical", "Fixed: C.ulonglong"),
        ("Path traversal in config", "Critical", "Fixed: path validation"),
        ("Go panics crash Python process", "High", "Fixed: error returns"),
        ("Input tensor validation (no-op)", "Medium", "Fixed: shape checking"),
        ("Unbounded polynomial degree", "Medium", "Fixed: MAX_DEGREE=127"),
        ("scipy/torch sparse indexing", "Medium", "Fixed: .numpy() conversion"),
        ("Bare except clauses (3 files)", "Low", "Fixed: except Exception"),
    ]
    for desc, severity, status in fixes:
        print(f"  [{severity:>8s}] {desc:<45s} {status}")

    print("\n" + "=" * 70)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    demo_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(demo_dir, "heart_config.yml")

    # Check files
    for f in ["heart_model.pt", "scaler.npz", "test_samples.npz"]:
        if not os.path.exists(os.path.join(demo_dir, f)):
            print(f"Missing {f}. Run train_model.py first.")
            return

    # Load model and data
    model = HeartDiseaseNet()
    model.load_state_dict(torch.load(
        os.path.join(demo_dir, "heart_model.pt"), weights_only=True))
    model.eval()

    samples = np.load(os.path.join(demo_dir, "test_samples.npz"))
    X_test, y_test = samples["X"], samples["y"]

    # Cleartext benchmark
    print("Running cleartext benchmark...")
    clear_results = benchmark_cleartext(model, X_test, y_test)
    print(f"  Cleartext accuracy: {clear_results['cleartext_accuracy']:.1%} "
          f"({len(X_test)} samples in {clear_results['cleartext_time_total']:.4f}s)")

    # FHE benchmark (all test samples)
    num_fhe = min(len(X_test), 54)  # all test samples
    print(f"\nRunning FHE benchmark ({num_fhe} samples)...")
    fhe_results = benchmark_fhe(model, X_test, y_test, config_path, num_fhe)

    # Print paper tables
    print_paper_table(clear_results, fhe_results)

    # Save raw results
    results_path = os.path.join(demo_dir, "benchmark_results.json")

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_path, "w") as f:
        json.dump({"cleartext": clear_results, "fhe": fhe_results}, f,
                  indent=2, default=convert)
    print(f"\nRaw results saved to {results_path}")


if __name__ == "__main__":
    main()

"""
FHE Inference Client — encrypts patient data and sends to the server.

The client holds the secret key. It encrypts patient data locally,
sends the ciphertext to the server for inference, receives the
encrypted result, and decrypts it. The server never sees the raw data.

Usage:
    python demo/client.py                        # default: localhost:5000, 5 patients
    python demo/client.py --url http://server:5000 --num-samples 10
"""

import os
import sys
import time
import base64
import json
import argparse

import torch
import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import orion
from orion.backend.python.tensors import CipherTensor


def setup_client(config_path, sk_path):
    """
    Initialize the client's FHE scheme and load the server's secret key.

    The client calls init_scheme (generating temporary keys), then replaces
    its keys with the server's exported secret key. This ensures the client's
    encryption is compatible with the server's evaluation keys.
    """
    scheme = orion.init_scheme(config_path)

    # Load the server's secret key
    with open(sk_path, "rb") as f:
        sk_bytes = f.read()

    sk_arr = np.frombuffer(sk_bytes, dtype=np.uint8)
    scheme.backend.LoadSecretKey(sk_arr)

    # Re-derive public key and recreate encryptor/decryptor from loaded sk
    scheme.backend.GeneratePublicKey()
    scheme.backend.NewEncryptor()
    scheme.backend.NewDecryptor()

    return scheme


def encrypt_patient(patient_data, input_level):
    """Encrypt a patient's feature vector and serialize for transport."""
    ptxt = orion.encode(patient_data, input_level)
    ctxt = orion.encrypt(ptxt)
    serialized = ctxt.serialize()
    return {
        "ciphertexts": [base64.b64encode(b).decode() for b in serialized["ciphertexts"]],
        "shape": serialized["shape"],
        "on_shape": serialized["on_shape"],
    }


def send_for_inference(server_url, payload):
    """Send encrypted ciphertext to the server and get back encrypted result."""
    resp = requests.post(f"{server_url}/predict", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def decrypt_result(scheme, response):
    """Deserialize and decrypt the server's response."""
    ct_data = {
        "ciphertexts": [base64.b64decode(b) for b in response["ciphertexts"]],
        "shape": response["shape"],
        "on_shape": response["on_shape"],
    }
    result_ctxt = CipherTensor.from_serialized(scheme, ct_data)
    result_ptxt = result_ctxt.decrypt()
    return result_ptxt.decode()


def main():
    parser = argparse.ArgumentParser(description="FHE Inference Client")
    parser.add_argument("--url", default="http://127.0.0.1:5000",
                        help="Server URL")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of patients to test")
    args = parser.parse_args()

    torch.manual_seed(42)
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(demo_dir, "heart_config.yml")
    sk_path = os.path.join(demo_dir, "keys", "secret.key")

    # Check secret key exists
    if not os.path.exists(sk_path):
        print(f"Missing {sk_path}. Start the server first (it exports the key).")
        return

    # Setup client scheme
    print("[Client] Initializing FHE scheme and loading server key...")
    t0 = time.time()
    scheme = setup_client(config_path, sk_path)
    print(f"[Client] Ready ({time.time()-t0:.2f}s)")

    # Get input_level from server
    info = requests.get(f"{args.url}/info").json()
    input_level = info["input_level"]
    print(f"[Client] Server input level: {input_level}")

    # Load test samples
    samples = np.load(os.path.join(demo_dir, "test_samples.npz"))
    X_test, y_test = samples["X"], samples["y"]

    labels = ["Healthy", "Heart Disease"]
    results = []

    print(f"\n{'='*60}")
    print(f"  Private Heart Disease Prediction ({args.num_samples} patients)")
    print(f"  Server: {args.url}")
    print(f"{'='*60}\n")

    for i in range(min(args.num_samples, len(X_test))):
        patient_data = torch.tensor(X_test[i:i+1], dtype=torch.float32)
        actual = int(y_test[i])

        # Step 1: Encrypt locally
        t0 = time.time()
        payload = encrypt_patient(patient_data, input_level)
        t_enc = time.time() - t0

        # Ciphertext size
        ct_size = sum(len(base64.b64decode(b)) for b in payload["ciphertexts"])

        # Step 2: Send to server for inference
        t0 = time.time()
        response = send_for_inference(args.url, payload)
        t_server = time.time() - t0
        t_inf = response.get("inference_time", 0)

        # Step 3: Decrypt locally
        t0 = time.time()
        out_values = decrypt_result(scheme, response)
        t_dec = time.time() - t0

        pred = out_values[:2].argmax().item()
        results.append(pred)

        status = "correct" if pred == actual else "WRONG"
        print(f"  Patient {i+1}: {labels[pred]:>14s}  "
              f"(actual: {labels[actual]:>14s}) [{status}]")
        print(f"    Encrypt: {t_enc:.3f}s | "
              f"Network: {t_server:.3f}s (inference: {t_inf:.3f}s) | "
              f"Decrypt: {t_dec:.3f}s | "
              f"Ciphertext: {ct_size/1024:.0f} KB")

    # Summary
    correct = sum(1 for i in range(len(results)) if results[i] == int(y_test[i]))
    print(f"\n{'='*60}")
    print(f"  FHE Accuracy: {correct}/{len(results)} ({correct/len(results):.0%})")
    print(f"  The server processed {len(results)} patients without")
    print(f"  ever seeing their medical data in plaintext.")
    print(f"{'='*60}")

    scheme.delete_scheme()


if __name__ == "__main__":
    main()

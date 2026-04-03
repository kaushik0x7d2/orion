"""
FHE Inference Server — runs encrypted inference on patient data.

The server holds the model weights and evaluation keys. It receives
encrypted ciphertexts from clients, runs inference, and returns
encrypted results. It NEVER sees the patient's raw data.

Usage:
    python demo/server.py          # starts on port 5000
    python demo/server.py --port 8080
"""

import os
import sys
import time
import base64
import json
import argparse

import torch
import numpy as np
from flask import Flask, request, jsonify
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import orion
from demo.train_model import HeartDiseaseNet

app = Flask(__name__)

# Global state set during startup
_state = {}


def startup(demo_dir):
    """Initialize the FHE scheme, load/compile model, export secret key."""
    config_path = os.path.join(demo_dir, "heart_config.yml")

    # Check required files
    for f in ["heart_model.pt", "scaler.npz", "test_samples.npz"]:
        path = os.path.join(demo_dir, f)
        if not os.path.exists(path):
            print(f"Missing {f}. Run train_model.py first.")
            sys.exit(1)

    # Load model
    model = HeartDiseaseNet()
    model.load_state_dict(torch.load(
        os.path.join(demo_dir, "heart_model.pt"), weights_only=True))
    model.eval()

    # Init FHE scheme
    print("[Server] Initializing FHE scheme...")
    t0 = time.time()
    scheme = orion.init_scheme(config_path)
    print(f"[Server] Scheme ready ({time.time()-t0:.2f}s)")

    # Export secret key for the client
    keys_dir = os.path.join(demo_dir, "keys")
    os.makedirs(keys_dir, exist_ok=True)
    sk_arr, sk_ptr = scheme.backend.SerializeSecretKey()
    sk_bytes = bytes(sk_arr)
    scheme.backend.FreeCArray(
        __import__('ctypes').cast(sk_ptr, __import__('ctypes').c_void_p))
    sk_path = os.path.join(keys_dir, "secret.key")
    with open(sk_path, "wb") as f:
        f.write(sk_bytes)
    print(f"[Server] Secret key exported to {sk_path} ({len(sk_bytes)} bytes)")

    # Fit model
    print("[Server] Fitting model for FHE...")
    t0 = time.time()
    samples = np.load(os.path.join(demo_dir, "test_samples.npz"))
    fit_X = torch.tensor(samples["X"], dtype=torch.float32)
    fit_dataset = TensorDataset(fit_X, torch.zeros(len(fit_X)))
    fit_loader = DataLoader(fit_dataset, batch_size=32)
    orion.fit(model, fit_loader)
    print(f"[Server] Fit done ({time.time()-t0:.2f}s)")

    # Compile
    print("[Server] Compiling model for FHE...")
    t0 = time.time()
    input_level = orion.compile(model)
    print(f"[Server] Compiled ({time.time()-t0:.2f}s) | Input level: {input_level}")

    # Switch to HE mode
    model.he()

    _state["model"] = model
    _state["scheme"] = scheme
    _state["input_level"] = input_level
    _state["config_path"] = config_path

    print("[Server] Ready to accept encrypted inference requests.")


@app.route("/info", methods=["GET"])
def info():
    """Return scheme parameters the client needs."""
    return jsonify({
        "input_level": _state["input_level"],
        "status": "ready",
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Run FHE inference on an encrypted ciphertext."""
    import gc
    from orion.backend.python.tensors import CipherTensor

    data = request.get_json()

    # Deserialize ciphertext
    ct_data = {
        "ciphertexts": [base64.b64decode(b) for b in data["ciphertexts"]],
        "shape": data["shape"],
        "on_shape": data["on_shape"],
    }
    ctxt = CipherTensor.from_serialized(_state["scheme"], ct_data)

    # Run encrypted inference (the model NEVER sees plaintext)
    t0 = time.time()
    out_ctxt = _state["model"](ctxt)
    t_inf = time.time() - t0

    # Serialize result ciphertext
    result = out_ctxt.serialize()
    response = {
        "ciphertexts": [base64.b64encode(b).decode() for b in result["ciphertexts"]],
        "shape": result["shape"],
        "on_shape": result["on_shape"],
        "inference_time": t_inf,
    }

    print(f"[Server] Inference completed in {t_inf:.3f}s")

    # Explicit cleanup to prevent Go heap buildup
    del ctxt, out_ctxt
    gc.collect()

    return jsonify(response)


def main():
    parser = argparse.ArgumentParser(description="FHE Inference Server")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    torch.manual_seed(42)
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    startup(demo_dir)

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()

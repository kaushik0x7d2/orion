"""
gRPC FHE Inference Client.

Encrypts patient data locally, sends raw binary ciphertexts to the
gRPC server (no base64 overhead), and decrypts results locally.

Usage:
    python demo/grpc_client.py                            # default server
    python demo/grpc_client.py --server localhost:50051 --num-samples 10

Requires:
    pip install grpcio grpcio-tools
"""

import os
import sys
import time
import argparse

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Try to import gRPC stubs
try:
    import grpc
    try:
        from demo import orion_fhe_pb2, orion_fhe_pb2_grpc
    except ImportError:
        demo_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, demo_dir)
        import orion_fhe_pb2
        import orion_fhe_pb2_grpc
except ImportError:
    print("grpcio not installed. pip install grpcio grpcio-tools")
    sys.exit(1)

import orion
from orion.backend.python.tensors import CipherTensor


def setup_client(config_path, sk_path):
    """Initialize scheme and load server's secret key."""
    scheme = orion.init_scheme(config_path)
    with open(sk_path, "rb") as f:
        sk_bytes = f.read()
    sk_arr = np.frombuffer(sk_bytes, dtype=np.uint8)
    scheme.backend.LoadSecretKey(sk_arr)
    scheme.backend.GeneratePublicKey()
    scheme.backend.NewEncryptor()
    scheme.backend.NewDecryptor()
    return scheme


def main():
    parser = argparse.ArgumentParser(description="gRPC FHE Client")
    parser.add_argument("--server", default="localhost:50051")
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(42)
    demo_dir = os.path.dirname(os.path.abspath(__file__))

    # Check prerequisites
    sk_path = os.path.join(demo_dir, "keys", "secret.key")
    config_path = os.path.join(demo_dir, "heart_config.yml")
    samples_path = os.path.join(demo_dir, "test_samples.npz")

    for p in [sk_path, config_path, samples_path]:
        if not os.path.exists(p):
            print(f"Missing: {p}")
            print("Start the gRPC server first (python demo/grpc_server.py)")
            return

    # Setup client scheme
    print("[Client] Setting up scheme...")
    scheme = setup_client(config_path, sk_path)

    # Connect to server
    print(f"[Client] Connecting to {args.server}...")
    channel = grpc.insecure_channel(
        args.server,
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ],
    )
    stub = orion_fhe_pb2_grpc.OrionFHEStub(channel)

    # Get server info
    info = stub.GetInfo(orion_fhe_pb2.InfoRequest())
    print(f"[Client] Server: {info.model_name} | Level: {info.input_level}")

    # Load test data
    samples = np.load(samples_path)
    X_test, y_test = samples["X"], samples["y"]
    labels = ["No Disease", "Disease"]
    num = min(args.num_samples, len(X_test))

    print(f"\n=== gRPC FHE Inference ({num} samples) ===\n")

    correct = 0
    total_rpc_time = 0

    for i in range(num):
        patient = torch.tensor(X_test[i:i+1], dtype=torch.float32)
        actual = int(y_test[i])

        # Encrypt locally
        t0 = time.time()
        ptxt = orion.encode(patient, info.input_level)
        ctxt = orion.encrypt(ptxt)
        serialized = ctxt.serialize()
        t_enc = time.time() - t0

        # Send via gRPC (raw bytes — no base64!)
        t0 = time.time()
        request = orion_fhe_pb2.PredictRequest(
            ciphertexts=serialized["ciphertexts"],
            shape=serialized["shape"],
            on_shape=serialized["on_shape"],
            request_id=f"sample-{i}",
        )
        response = stub.Predict(request)
        t_rpc = time.time() - t0
        total_rpc_time += t_rpc

        if response.error:
            print(f"  Sample {i+1}: ERROR — {response.error}")
            continue

        # Deserialize and decrypt locally
        t0 = time.time()
        result_data = {
            "ciphertexts": [bytes(ct) for ct in response.ciphertexts],
            "shape": list(response.shape),
            "on_shape": list(response.on_shape),
        }
        result_ctxt = CipherTensor.from_serialized(scheme, result_data)
        result = result_ctxt.decrypt().decode()
        t_dec = time.time() - t0

        pred = result.flatten()[:2].argmax().item()
        if pred == actual:
            correct += 1
        status = "ok" if pred == actual else "WRONG"

        ct_size_kb = sum(len(ct) for ct in serialized["ciphertexts"]) / 1024

        print(f"  Sample {i+1:2d}/{num}: {labels[pred]:>12s} "
              f"(actual: {labels[actual]:>12s}) [{status:>5s}] "
              f"| enc={t_enc:.2f}s rpc={t_rpc:.2f}s dec={t_dec:.2f}s "
              f"| server_inf={response.inference_time_seconds:.2f}s "
              f"| {ct_size_kb:.0f}KB")

    print(f"\n=== Summary ===")
    print(f"  Accuracy: {correct}/{num} ({correct/num:.1%})")
    print(f"  Avg RPC time: {total_rpc_time/num:.2f}s")
    print(f"  Protocol: gRPC (binary, no base64 overhead)")

    # Health check
    health = stub.HealthCheck(orion_fhe_pb2.HealthRequest())
    print(f"  Server health: {health.status} | "
          f"Uptime: {health.uptime_seconds:.0f}s | "
          f"Requests served: {health.requests_served}")

    channel.close()
    scheme.delete_scheme()


if __name__ == "__main__":
    main()

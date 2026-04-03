"""
gRPC FHE Inference Server.

Production-grade alternative to the Flask REST server. Uses gRPC with
Protocol Buffers for efficient binary ciphertext transport (no base64
overhead), streaming batch inference, and built-in health checks.

Usage:
    python demo/grpc_server.py                    # port 50051
    python demo/grpc_server.py --port 50051 --workers 4

Requires:
    pip install grpcio grpcio-tools
    python -m grpc_tools.protoc -Idemo --python_out=demo --grpc_python_out=demo demo/orion_fhe.proto
"""

import os
import sys
import gc
import time
import argparse
import logging
from concurrent import futures

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger("orion.grpc_server")

# Try to import gRPC — generate stubs if needed
try:
    import grpc
    # Try importing generated stubs
    try:
        import orion_fhe_pb2
        import orion_fhe_pb2_grpc
    except ImportError:
        # Auto-generate from .proto
        logger.info("Generating gRPC stubs from orion_fhe.proto...")
        from grpc_tools import protoc
        demo_dir = os.path.dirname(os.path.abspath(__file__))
        protoc.main([
            'grpc_tools.protoc',
            f'-I{demo_dir}',
            f'--python_out={demo_dir}',
            f'--grpc_python_out={demo_dir}',
            os.path.join(demo_dir, 'orion_fhe.proto'),
        ])
        import orion_fhe_pb2
        import orion_fhe_pb2_grpc
except ImportError:
    grpc = None
    logger.warning("grpcio not installed. Install with: pip install grpcio grpcio-tools")

import orion
from orion.core.memory import get_memory_stats, cleanup_all
from orion.core.crypto_utils import CiphertextAuthenticator


class OrionFHEServicer:
    """gRPC service implementation for FHE inference."""

    def __init__(self, model, scheme, input_level, model_name="HeartDiseaseNet",
                 hmac_key=None):
        self.model = model
        self.scheme = scheme
        self.input_level = input_level
        self.model_name = model_name
        self.start_time = time.time()
        self.request_count = 0
        self.authenticator = (
            CiphertextAuthenticator(hmac_key) if hmac_key else None
        )

    def GetInfo(self, request, context):
        return orion_fhe_pb2.InfoResponse(
            input_level=self.input_level,
            model_name=self.model_name,
            status="serving",
            num_features=13,
            num_classes=2,
        )

    def Predict(self, request, context):
        """Run FHE inference on a single encrypted sample."""
        from orion.backend.python.tensors import CipherTensor

        request_id = request.request_id or f"req-{self.request_count}"
        self.request_count += 1

        try:
            # Verify HMAC if authenticator is configured
            if self.authenticator and request.hmac:
                payload = {
                    "ciphertexts": list(request.ciphertexts),
                    "hmac": request.hmac,
                }
                if not self.authenticator.verify(payload):
                    context.set_code(grpc.StatusCode.UNAUTHENTICATED)
                    context.set_details("HMAC verification failed")
                    return orion_fhe_pb2.PredictResponse(
                        error="HMAC verification failed",
                        request_id=request_id,
                    )

            # Deserialize ciphertext from raw bytes (no base64!)
            ct_data = {
                "ciphertexts": [bytes(ct) for ct in request.ciphertexts],
                "shape": list(request.shape),
                "on_shape": list(request.on_shape),
            }
            ctxt = CipherTensor.from_serialized(self.scheme, ct_data)

            # Run encrypted inference
            t0 = time.time()
            out_ctxt = self.model(ctxt)
            t_inf = time.time() - t0

            # Serialize result
            result = out_ctxt.serialize()

            # Build response
            response = orion_fhe_pb2.PredictResponse(
                ciphertexts=result["ciphertexts"],
                shape=result["shape"],
                on_shape=result["on_shape"],
                inference_time_seconds=t_inf,
                request_id=request_id,
            )

            # Sign response if authenticator configured
            if self.authenticator:
                signed = self.authenticator.sign(result)
                response.hmac = signed["hmac"]

            logger.info("[%s] Inference: %.3fs", request_id, t_inf)

            # Cleanup
            del ctxt, out_ctxt
            gc.collect()

            return response

        except Exception as e:
            logger.error("[%s] Error: %s", request_id, e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return orion_fhe_pb2.PredictResponse(
                error=str(e),
                request_id=request_id,
            )

    def PredictBatch(self, request_iterator, context):
        """Stream multiple predictions."""
        for request in request_iterator:
            yield self.Predict(request, context)

    def HealthCheck(self, request, context):
        stats = get_memory_stats(self.scheme.backend)
        return orion_fhe_pb2.HealthResponse(
            status="serving",
            live_ciphertexts=stats["ciphertexts"],
            live_plaintexts=stats["plaintexts"],
            uptime_seconds=time.time() - self.start_time,
            requests_served=self.request_count,
        )


def create_server(model, scheme, input_level, port=50051, max_workers=4,
                  model_name="HeartDiseaseNet", hmac_key=None):
    """Create and configure a gRPC server."""
    if grpc is None:
        raise ImportError("grpcio not installed. pip install grpcio grpcio-tools")

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ],
    )

    servicer = OrionFHEServicer(
        model, scheme, input_level, model_name, hmac_key)
    orion_fhe_pb2_grpc.add_OrionFHEServicer_to_server(servicer, server)

    server.add_insecure_port(f"[::]:{port}")
    return server


def startup(demo_dir):
    """Initialize scheme, load and compile model."""
    from torch.utils.data import TensorDataset, DataLoader
    from demo.train_model import HeartDiseaseNet

    config_path = os.path.join(demo_dir, "heart_config.yml")

    for f in ["heart_model.pt", "scaler.npz", "test_samples.npz"]:
        path = os.path.join(demo_dir, f)
        if not os.path.exists(path):
            print(f"Missing {f}. Run train_model.py first.")
            sys.exit(1)

    model = HeartDiseaseNet()
    model.load_state_dict(torch.load(
        os.path.join(demo_dir, "heart_model.pt"), weights_only=True))
    model.eval()

    print("[gRPC Server] Initializing FHE scheme...")
    t0 = time.time()
    scheme = orion.init_scheme(config_path)
    print(f"[gRPC Server] Scheme ready ({time.time()-t0:.2f}s)")

    # Export secret key
    import ctypes
    keys_dir = os.path.join(demo_dir, "keys")
    os.makedirs(keys_dir, exist_ok=True)
    sk_arr, sk_ptr = scheme.backend.SerializeSecretKey()
    sk_bytes = bytes(sk_arr)
    scheme.backend.FreeCArray(ctypes.cast(sk_ptr, ctypes.c_void_p))
    sk_path = os.path.join(keys_dir, "secret.key")
    with open(sk_path, "wb") as f:
        f.write(sk_bytes)

    # Fit
    print("[gRPC Server] Fitting model...")
    t0 = time.time()
    samples = np.load(os.path.join(demo_dir, "test_samples.npz"))
    fit_X = torch.tensor(samples["X"], dtype=torch.float32)
    fit_dataset = TensorDataset(fit_X, torch.zeros(len(fit_X)))
    orion.fit(model, DataLoader(fit_dataset, batch_size=32))
    print(f"[gRPC Server] Fit done ({time.time()-t0:.2f}s)")

    # Compile
    print("[gRPC Server] Compiling...")
    t0 = time.time()
    input_level = orion.compile(model)
    print(f"[gRPC Server] Compiled ({time.time()-t0:.2f}s) | Level: {input_level}")

    model.he()
    return model, scheme, input_level


def main():
    parser = argparse.ArgumentParser(description="gRPC FHE Inference Server")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    torch.manual_seed(42)
    demo_dir = os.path.dirname(os.path.abspath(__file__))

    model, scheme, input_level = startup(demo_dir)
    server = create_server(model, scheme, input_level, port=args.port,
                           max_workers=args.workers)
    server.start()

    print(f"\n[gRPC Server] Listening on port {args.port}")
    print(f"[gRPC Server] Workers: {args.workers}")
    print("[gRPC Server] Press Ctrl+C to stop.\n")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[gRPC Server] Shutting down...")
        server.stop(grace=5)
        scheme.delete_scheme()


if __name__ == "__main__":
    main()

# Orion — Hardened FHE Framework for Private Neural Inference

A production-hardened version of [Orion](https://github.com/baahl-nyu/orion) (ASPLOS'25 Best Paper), a PyTorch-integrated Fully Homomorphic Encryption framework that enables neural network inference on encrypted data.

**What's new in v1.1.0:** 8 vulnerability fixes (including a critical Windows portability bug), 10 production features (HuggingFace conversion, FHE transformer layers), 66 automated tests, and 4 inference demos. See [HARDENING.md](HARDENING.md) for the full technical report.

---

## Key Features

- **Private inference** — Run neural networks on encrypted patient data. The server never sees plaintext.
- **PyTorch integration** — Define models with `orion.nn` (drop-in replacements for `torch.nn`), then compile for FHE.
- **Security validated** — CKKS parameters checked against HE Standard bounds before any computation.
- **HuggingFace compatible** — Convert standard `torch.nn` models to FHE with `convert_to_orion()`. Automatic compatibility checking.
- **Transformer support** — PolySoftmax (PowerSoftmax, ICLR'25), FHELayerNorm, FHEMultiHeadAttention for encrypted transformer inference.
- **Production-ready** — Memory management, caching, error propagation, gRPC transport, HMAC auth, key encryption.
- **Cross-platform** — Fixed critical Windows portability bug (`c_ulong` overflow). Works on Windows and Linux.

## Quick Start

### Prerequisites

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.9 – 3.13 | Framework runtime |
| Go | 1.21+ | Build Lattigo backend |
| GCC/MinGW | Any recent | CGo compilation |

### Install

```bash
git clone <this-repo>
cd orion

# Install Python dependencies
pip install -e ".[all]"

# Build the Go backend (one-time)
cd orion/backend/lattigo
go build -buildmode=c-shared -o lattigo-linux.so .       # Linux
go build -buildmode=c-shared -o lattigo-windows.dll .     # Windows
cd ../../..
```

### Run FHE Inference Demo

```bash
# 1. Train a model (HeartDiseaseNet, ~5 seconds)
python demo/train_model.py

# 2. Run encrypted inference (5 patients, ~60 seconds total)
python demo/fhe_inference.py
```

Expected output:
```
=== Cleartext Inference (5 samples) ===
  Patient 1/5: Healthy    (actual: Healthy)    [   ok]
  ...

=== FHE Inference (5 samples) ===
  Patient 1/5: Healthy    (actual: Healthy)    [   ok] | 5.35s | 1.8 bits
  Patient 2/5: Healthy    (actual: Healthy)    [   ok] | 5.39s | 1.8 bits
  ...

FHE Accuracy: 5/5 (100%)
```

### Run Tests

```bash
pytest tests/ -v
# 66 passed in 9.8s
```

---

## Architecture

```
Client                          Server
┌──────────┐                    ┌──────────────────┐
│ encrypt  │───ciphertext──────>│ FHE inference    │
│ (local)  │                    │ (never sees data)│
│          │<──ciphertext───────│                  │
│ decrypt  │                    │ model weights +  │
│ (local)  │                    │ evaluation keys  │
└──────────┘                    └──────────────────┘
```

## Usage

### Basic FHE Inference

```python
import torch
import orion
from orion.nn import Linear, SiLU, Module

# Define model using orion.nn (same API as torch.nn)
class MyModel(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(13, 64)
        self.act1 = SiLU(degree=7)
        self.fc2 = Linear(64, 2)

    def forward(self, x):
        return self.fc2(self.act1(self.fc1(x)))

# Standard PyTorch training...
model = MyModel()
# train(model, data)

# Prepare for FHE
scheme = orion.init_scheme("config.yml")
orion.fit(model, dataloader)
input_level = orion.compile(model)

# Encrypt → Infer → Decrypt
ptxt = orion.encode(patient_data, input_level)
ctxt = orion.encrypt(ptxt)
model.he()
result = model(ctxt)
output = result.decrypt().decode()
```

### Convert HuggingFace / PyTorch Models

```python
import torch.nn as nn
from orion.integrations import convert_to_orion, check_compatibility

# Any standard PyTorch model — no Orion imports needed for training
model = nn.Sequential(
    nn.Linear(13, 64), nn.SiLU(),
    nn.Linear(64, 32), nn.SiLU(),
    nn.Linear(32, 2),
)
# ... train as usual with torch.optim ...

# Check FHE compatibility
report = check_compatibility(model)
print(report)  # Shows supported/unsupported layers, estimated depth

# Convert to Orion (replaces layers, transfers weights)
orion_model = convert_to_orion(model, activation_degree=7)

# Now use with the standard Orion FHE pipeline:
# orion.fit(orion_model, dataloader)
# orion.compile(orion_model)
# model.he() → encrypt → infer → decrypt
```

Supported: Linear, Conv2d, BatchNorm, LayerNorm, AvgPool, Flatten, ReLU, GELU, SiLU, Sigmoid, ELU, SELU, Mish, Softplus. Dropout is auto-removed. MaxPool, Softmax, Attention, RNN/LSTM are flagged as incompatible.

### FHE-Compatible Transformers

```python
from orion.nn.transformer import FHETransformerEncoderLayer

# PolySoftmax replaces exp-softmax with x^p / sum(x^p)
# Proven at 1.4B parameter scale (PowerSoftmax, ICLR 2025)
encoder = FHETransformerEncoderLayer(
    d_model=64, nhead=4, dim_feedforward=128,
    softmax_power=4,       # PolySoftmax with p=4
    activation_degree=7,   # Chebyshev GELU approximation
)

x = torch.randn(8, 16, 64)   # [batch, seq_len, d_model]
out = encoder(x)              # same shape, FHE-compatible
```

### Client-Server Deployment

```bash
# REST (simple)
python demo/server.py --port 5000
python demo/client.py --url http://127.0.0.1:5000

# gRPC (production — 33% less bandwidth)
pip install grpcio grpcio-tools
python demo/grpc_server.py --port 50051
python demo/grpc_client.py --server localhost:50051
```

### Production Features

```python
import orion

# Validate security before computing
orion.validate_ckks_params(logn=14, logq=[...], logp=[...], logscale=35)

# Memory management
with orion.managed_cipher(encrypted_data) as ct:
    result = model(ct)  # Go-side memory freed on exit

# Cache compiled models (skip 60s fit+compile on restart)
cache = orion.FHECache()
cache.save(config, model, input_level)

# Authenticate ciphertexts in transit
auth = orion.CiphertextAuthenticator.from_secret_key(sk_bytes)
signed = auth.sign(ctxt.serialize())

# Encrypt secret keys at rest
enc = orion.KeyEncryptor(password="strong-passphrase")
enc.encrypt_to_file(sk_bytes, "secret.key.enc")
```

---

## Demos

| Demo | Model | Accuracy | FHE Time | Command |
|------|-------|----------|----------|---------|
| Heart Disease | 13→64→32→2 (SiLU) | 85% clear, 100% FHE* | 5.1s/patient | `python demo/fhe_inference.py` |
| Breast Cancer | 30→128→64→2 (GELU) | 97% clear, 95% FHE | 16s/patient | `python demo/cancer_fhe_inference.py` |
| Client-Server | Heart Disease | 5/5 over HTTP | 5s + network | `python demo/server.py` + `client.py` |
| Benchmark | Heart Disease (54 samples) | 81.5% FHE | 5.14s avg | `python demo/benchmark.py` |
| HuggingFace | Iris MLP (SiLU) | 93% clear, 80% FHE | 2.9s/sample | `python demo/huggingface_demo.py` |
| Transformer | Iris PolySoftmax p=4 | 89% cleartext | N/A (cleartext) | `python demo/transformer_demo.py` |

*On 5-sample subset. Full 54-sample accuracy is 81.5%.

---

## Security

### Vulnerabilities Fixed

| Severity | Issue | Impact |
|----------|-------|--------|
| **Critical** | `c_ulong` 32-bit overflow on Windows | Silent data corruption for LogScale > 31 |
| **Critical** | Go panics crash Python process | Unrecoverable crashes during inference |
| **High** | Path traversal in config files | Arbitrary file read/delete |
| **High** | No FFI error propagation | Silent failures, wrong results |
| **Medium** | No input validation, unbounded degrees, bare excepts | Various |

### Security Guarantees

- CKKS parameters validated against HE Standard (rejects < 128-bit security)
- All Go→Python errors propagated automatically (no silent failures)
- Ciphertext authentication (HMAC-SHA256) for tamper detection
- Secret key encryption at rest (AES-256-GCM, PBKDF2 600K iterations)
- Thread-safe: 8 Go mutexes (152 lock/unlock calls) + Python Lock
- Path traversal prevention on all file I/O

---

## Project Structure

```
orion/
├── core/
│   ├── orion.py              # Main Scheme class (fit, compile, infer)
│   ├── config_validator.py   # HE Standard parameter validation
│   ├── error_handling.py     # FFI error propagation
│   ├── memory.py             # Memory management & tracking
│   ├── parallel.py           # Batch parallel inference
│   ├── cache.py              # Compiled model caching
│   └── crypto_utils.py       # HMAC auth & key encryption
├── backend/
│   ├── lattigo/              # Go FFI backend (CGo → Lattigo v6)
│   │   ├── *.go              # 8 Go files with mutexes + error handling
│   │   └── bindings.py       # Python ctypes bridge
│   └── python/               # Python-side wrappers
│       ├── parameters.py     # Config parsing + validation
│       └── tensors.py        # PlainTensor / CipherTensor
├── nn/                       # FHE-compatible neural network layers
│   └── transformer.py        # PolySoftmax, FHELayerNorm, FHE attention
├── integrations/
│   └── huggingface.py        # PyTorch/HuggingFace → Orion converter
demo/
├── train_model.py            # Heart disease model training
├── train_cancer.py           # Breast cancer model training
├── fhe_inference.py          # End-to-end FHE demo
├── server.py / client.py     # REST client-server
├── grpc_server.py / grpc_client.py  # gRPC (production)
├── benchmark.py              # Paper-ready benchmarks
├── huggingface_demo.py       # HuggingFace model conversion demo
└── transformer_demo.py       # FHE-compatible transformer demo
tests/
└── test_hardening.py         # 66 automated tests
```

---

## Citation

This work builds on the Orion framework:

```bibtex
@inproceedings{ebel2025orion,
  title={Orion: A Fully Homomorphic Encryption Compiler for Private Deep Learning},
  author={Ebel, Austin and Ozsoy, Karthik and Yigitoglu, Enes and Cammarota, Rosario and Maniatakos, Michail},
  booktitle={Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS'25)},
  year={2025}
}
```

## License

See the [original Orion repository](https://github.com/baahl-nyu/orion) for license terms.

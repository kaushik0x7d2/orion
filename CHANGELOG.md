# Changelog

## v1.1.0 — Hardened Release (2026-04-04)

### Critical Fixes
- **Windows portability bug**: Fixed `C.ulong`/`ctypes.c_ulong` 32-bit overflow on Windows causing silent data corruption for LogScale > 31. Changed all scale-related FFI parameters to `C.ulonglong`/`ctypes.c_ulonglong` (64-bit).
- **Go panic handling**: Replaced all `panic()` calls in 8 Go files with `lastError` error reporting + sentinel return values. Python process no longer crashes on Go-side errors.

### Security Fixes
- Path traversal prevention: `_validate_path()` in `parameters.py` confines file paths to the working directory (only enforced in save/load I/O mode).
- FFI error propagation: `OrionGetLastError`/`OrionClearLastError` + automatic `wrap_ffi_error_checking()` on all Go calls.
- Input validation: Tensor ID count mismatch checking in `_check_valid()`.
- Polynomial degree bounds: `MAX_CHEBYSHEV_DEGREE = 127` in `activation.py`.
- Bare except clauses: Changed `except:` to `except Exception:` in 3 files.
- scipy/torch interop: Added `.numpy()` conversions in `packing.py`.

### New Features
- **Config validation** (`orion/core/config_validator.py`): Validates CKKS parameters against HE Standard security tables. Rejects insecure parameters before computation.
- **Error decorator** (`orion/core/error_handling.py`): `@check_ffi_error` and `wrap_ffi_error_checking()` auto-wrap Go FFI calls with error checking.
- **Memory management** (`orion/core/memory.py`): Context managers (`managed_cipher`/`managed_plain`), `MemoryTracker`, `cleanup_all()`.
- **Batch processing** (`orion/core/parallel.py`): `PipelineExecutor` for threaded pipeline parallelism, `BatchProcessor` for memory-safe sequential processing.
- **Model caching** (`orion/core/cache.py`): `FHECache` saves/loads compiled model state to skip expensive fit+compile.
- **Crypto utilities** (`orion/core/crypto_utils.py`): `CiphertextAuthenticator` (HMAC-SHA256) and `KeyEncryptor` (AES-256-GCM with PBKDF2).
- **gRPC server** (`demo/grpc_server.py`): Binary ciphertext transport (33% less bandwidth than REST+base64), streaming batch inference, health checks.
- **Thread safety**: 8 Go mutexes (152 lock/unlock calls) + Python `threading.Lock`.
- **Ciphertext serialization**: `SerializeCiphertext`/`LoadCiphertext` in Go + Python `serialize()`/`from_serialized()`.
- **HuggingFace integration** (`orion/integrations/huggingface.py`): Convert standard PyTorch models to FHE-compatible Orion models. Compatibility checker analyzes model layers. Supports Linear, Conv2d, BatchNorm, LayerNorm, AvgPool, 8 activation functions. Automatic dropout removal and weight transfer.
- **FHE Transformer layers** (`orion/nn/transformer.py`): PolySoftmax (x^p/sum(x^p), PowerSoftmax ICLR'25), FHELayerNorm, FHEMultiHeadAttention, FHETransformerEncoderLayer. Enables encrypted inference on transformer-based models.

### Demo Applications
- Heart disease diagnosis (HeartDiseaseNet, SiLU, 85% cleartext → 100% FHE on 5 samples)
- Breast cancer diagnosis (BreastCancerNet, GELU, 97% cleartext → 95% FHE on 20 samples)
- REST client-server (Flask)
- gRPC client-server (production)
- Paper-ready benchmarks
- HuggingFace model conversion (Iris MLP → FHE, compatibility checker, activation variety)
- Transformer FHE demo (PolySoftmax training, compatibility analysis, LayerNorm conversion)

### Testing
- 66 automated tests (all passing on Windows 11 / Python 3.13)
- Config validation, path traversal, error handling, memory management, HMAC auth, key encryption, polynomial bounds, model caching, Windows portability, HuggingFace integration, transformer layers

### Other Changes
- Version bumped to 1.1.0
- Python compatibility extended to 3.9–3.13
- Added `cryptography` dependency for AES-256-GCM
- Added optional dependency groups: `[server]`, `[grpc]`, `[all]`, `[dev]`
- Updated `.gitignore` for demo artifacts and cache

## v1.0.2 — Original Release (ASPLOS'25)

Original Orion framework as published in the ASPLOS'25 paper.

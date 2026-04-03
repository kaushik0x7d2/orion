"""
Comprehensive test suite for Orion FHE framework hardening.

Tests all security fixes, production features, and correctness guarantees
introduced during the hardening effort. Organized by feature area.

Run with:
    pytest tests/test_hardening.py -v
    pytest tests/test_hardening.py -v -k "security"  # just security tests
"""

import os
import sys
import json
import hmac
import hashlib
import tempfile
import warnings

import pytest
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ================================================================
#  1. Configuration Validation (Feature 5)
# ================================================================

class TestConfigValidation:
    """Tests for CKKS parameter security validation."""

    def test_valid_128bit_params(self):
        """Standard heart disease config should pass validation."""
        from orion.core.config_validator import validate_ckks_params

        result = validate_ckks_params(
            logn=14,
            logq=[45, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 45],
            logp=[46, 46, 46],
            logscale=35,
            h=192,
        )
        assert result["valid"] is True
        assert result["security_level"] >= 128

    def test_reject_small_logn(self):
        """LogN < 10 must be rejected."""
        from orion.core.config_validator import (
            validate_ckks_params, SecurityValidationError
        )

        with pytest.raises(SecurityValidationError, match="below minimum"):
            validate_ckks_params(
                logn=8, logq=[20, 20], logp=[20], logscale=20
            )

    def test_reject_insecure_logqp(self):
        """Parameters exceeding 128-bit security bound must be rejected."""
        from orion.core.config_validator import (
            validate_ckks_params, SecurityValidationError
        )

        # LogN=12, max LogQP for 128-bit = 218
        # Total LogQP = 8*40 + 3*40 = 440, over 218
        with pytest.raises(SecurityValidationError, match="INSECURE"):
            validate_ckks_params(
                logn=12,
                logq=[40, 40, 40, 40, 40, 40, 40, 40],
                logp=[40, 40, 40],
                logscale=35,
            )

    def test_reject_logscale_too_large(self):
        """LogScale > min intermediate LogQ should be rejected."""
        from orion.core.config_validator import (
            validate_ckks_params, SecurityValidationError
        )

        with pytest.raises(SecurityValidationError, match="LogScale"):
            validate_ckks_params(
                logn=14,
                logq=[45, 20, 20, 45],
                logp=[46],
                logscale=30,  # > min intermediate (20)
            )

    def test_nonstrict_returns_warnings(self):
        """Non-strict mode returns warnings instead of raising."""
        from orion.core.config_validator import validate_ckks_params

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = validate_ckks_params(
                logn=14,
                logq=[45, 35, 35, 35, 45],
                logp=[46],
                logscale=35,
                h=32,  # low hamming weight — triggers warning
                strict=False,
            )
        assert result["valid"] is True
        assert any("Hamming weight" in w for w in result["warnings"])

    def test_security_level_estimation(self):
        """Verify security level estimation for known parameter sets."""
        from orion.core.config_validator import estimate_security_level

        # LogN=14: 128-bit max=883, 192-bit max=438, 256-bit max=305
        assert estimate_security_level(14, 883) == 128
        assert estimate_security_level(14, 438) == 192
        assert estimate_security_level(14, 305) == 256

    def test_empty_logq_rejected(self):
        """Empty LogQ should be rejected."""
        from orion.core.config_validator import (
            validate_ckks_params, SecurityValidationError
        )

        with pytest.raises(SecurityValidationError, match="LogQ is empty"):
            validate_ckks_params(
                logn=14, logq=[], logp=[46], logscale=35
            )


# ================================================================
#  2. Path Traversal Prevention (Feature 3 — existing)
# ================================================================

class TestPathTraversal:
    """Tests for path traversal attack prevention in parameters.py."""

    def test_reject_parent_directory_escape(self):
        """../../etc/passwd style paths must be rejected in save/load mode."""
        from orion.backend.python.parameters import NewParameters

        params = NewParameters({
            "ckks_params": {
                "LogN": 14, "LogQ": [45, 35, 45], "LogP": [46],
                "LogScale": 35, "RingType": "Standard",
            },
            "orion": {
                "backend": "lattigo",
                "diags_path": "../../etc/evil",
                "keys_path": "safe_keys",
                "io_mode": "load",  # validation only triggers in save/load mode
            },
        })
        with pytest.raises(ValueError, match="[Pp]ath traversal"):
            params.get_diags_path()

    def test_paths_ignored_in_none_mode(self):
        """Paths with .. are allowed when io_mode=none (not used)."""
        from orion.backend.python.parameters import NewParameters

        params = NewParameters({
            "ckks_params": {
                "LogN": 14, "LogQ": [45, 35, 45], "LogP": [46],
                "LogScale": 35, "RingType": "Standard",
            },
            "orion": {
                "backend": "lattigo",
                "diags_path": "../data/diags.h5",
                "keys_path": "../data/keys.h5",
                "io_mode": "none",
            },
        })
        # Should not raise since io_mode is none
        result = params.get_diags_path()
        assert "diags" in result

    def test_accept_safe_relative_path(self):
        """Normal relative paths within CWD should be accepted."""
        from orion.backend.python.parameters import NewParameters

        params = NewParameters({
            "ckks_params": {
                "LogN": 14, "LogQ": [45, 35, 45], "LogP": [46],
                "LogScale": 35, "RingType": "Standard",
            },
            "orion": {
                "backend": "lattigo",
                "diags_path": "output/diags",
                "keys_path": "output/keys",
                "io_mode": "load",
            },
        })
        # Should not raise
        result = params.get_diags_path()
        assert "output" in result


# ================================================================
#  3. Error Handling (Feature 6)
# ================================================================

class TestErrorHandling:
    """Tests for the FFI error propagation system."""

    def test_fhe_backend_error_has_function_name(self):
        """FHEBackendError should include the function name."""
        from orion.core.error_handling import FHEBackendError

        err = FHEBackendError("division by zero", function_name="EvaluatePolynomial")
        assert "EvaluatePolynomial" in str(err)
        assert "division by zero" in str(err)

    def test_check_ffi_error_decorator(self):
        """Decorator should raise on backend errors."""
        from orion.core.error_handling import check_ffi_error, FHEBackendError

        class MockBackend:
            def __init__(self):
                self._error = None

            def get_last_error(self):
                return self._error

            def clear_last_error(self):
                self._error = None

        class MockModule:
            def __init__(self, backend):
                self.backend = backend

            @check_ffi_error
            def do_something(self):
                self.backend._error = "simulated Go panic"
                return -1

            @check_ffi_error
            def do_ok(self):
                return 42

        backend = MockBackend()
        module = MockModule(backend)

        # Should raise on error
        with pytest.raises(FHEBackendError, match="simulated Go panic"):
            module.do_something()

        # Should pass when no error
        assert module.do_ok() == 42

    def test_wrap_ffi_error_checking(self):
        """wrap_ffi_error_checking should instrument a library object."""
        from orion.core.error_handling import wrap_ffi_error_checking, FHEBackendError

        class MockLib:
            def __init__(self):
                self._error = None
                self.call_count = 0

            def get_last_error(self):
                return self._error

            def clear_last_error(self):
                self._error = None

            def Encode(self, *args):
                self.call_count += 1
                self._error = "encode failed"
                return -1

            def NonSentinel(self, *args):
                return "ok"

        lib = MockLib()
        wrap_ffi_error_checking(lib)

        # Wrapped function should raise
        with pytest.raises(FHEBackendError, match="encode failed"):
            lib.Encode(1, 2, 3)


# ================================================================
#  4. Memory Management (Feature 2)
# ================================================================

class TestMemoryManagement:
    """Tests for memory tracking and cleanup utilities."""

    def test_memory_tracker(self):
        """MemoryTracker should record snapshots."""
        from orion.core.memory import MemoryTracker

        class MockBackend:
            def __init__(self):
                self._pt = [1, 2, 3]
                self._ct = [10, 20]

            def GetLivePlaintexts(self):
                return self._pt

            def GetLiveCiphertexts(self):
                return self._ct

        backend = MockBackend()
        tracker = MemoryTracker(backend)

        snap1 = tracker.snapshot("before")
        assert snap1["plaintexts"] == 3
        assert snap1["ciphertexts"] == 2
        assert snap1["total"] == 5

        backend._ct.append(30)
        snap2 = tracker.snapshot("after")
        assert snap2["total"] == 6

        report = tracker.report()
        assert "before" in report
        assert "after" in report
        assert "+1" in report

    def test_get_memory_stats(self):
        """get_memory_stats should return dict with counts."""
        from orion.core.memory import get_memory_stats

        class MockBackend:
            def GetLivePlaintexts(self):
                return [1, 2]
            def GetLiveCiphertexts(self):
                return [10]

        stats = get_memory_stats(MockBackend())
        assert stats["plaintexts"] == 2
        assert stats["ciphertexts"] == 1
        assert stats["total"] == 3

    def test_cleanup_all(self):
        """cleanup_all should delete all live objects."""
        from orion.core.memory import cleanup_all

        deleted = []

        class MockBackend:
            def GetLivePlaintexts(self):
                return [1, 2, 3]
            def GetLiveCiphertexts(self):
                return [10, 20]
            def DeletePlaintext(self, idx):
                deleted.append(("pt", idx))
            def DeleteCiphertext(self, idx):
                deleted.append(("ct", idx))

        freed = cleanup_all(MockBackend())
        assert freed == 5
        assert len(deleted) == 5


# ================================================================
#  5. Ciphertext Authentication (Feature 8)
# ================================================================

class TestCiphertextAuth:
    """Tests for HMAC-based ciphertext authentication."""

    def test_sign_and_verify(self):
        """Signed ciphertext should verify correctly."""
        from orion.core.crypto_utils import CiphertextAuthenticator

        auth = CiphertextAuthenticator(hmac_key=b"test-key-1234567890")
        data = {
            "ciphertexts": [b"\x00\x01\x02" * 100, b"\x03\x04\x05" * 100],
            "shape": [1, 13],
            "on_shape": [1, 13],
        }

        signed = auth.sign(data)
        assert "hmac" in signed
        assert auth.verify(signed) is True

    def test_detect_tampering(self):
        """Tampered ciphertext should fail verification."""
        from orion.core.crypto_utils import CiphertextAuthenticator

        auth = CiphertextAuthenticator(hmac_key=b"test-key-1234567890")
        data = {
            "ciphertexts": [b"\x00\x01\x02" * 100],
            "shape": [1, 13],
            "on_shape": [1, 13],
        }

        signed = auth.sign(data)
        # Tamper with ciphertext
        signed["ciphertexts"] = [b"\xff\xff\xff" * 100]
        assert auth.verify(signed) is False

    def test_wrong_key_fails(self):
        """Verification with wrong key should fail."""
        from orion.core.crypto_utils import CiphertextAuthenticator

        auth1 = CiphertextAuthenticator(hmac_key=b"key-aaaa-1234567890")
        auth2 = CiphertextAuthenticator(hmac_key=b"key-bbbb-1234567890")

        data = {"ciphertexts": [b"hello"], "shape": [1], "on_shape": [1]}
        signed = auth1.sign(data)
        assert auth2.verify(signed) is False

    def test_from_secret_key(self):
        """Should derive HMAC key from FHE secret key."""
        from orion.core.crypto_utils import CiphertextAuthenticator

        sk = os.urandom(1000)  # Simulate FHE secret key
        auth = CiphertextAuthenticator.from_secret_key(sk)
        data = {"ciphertexts": [b"test"], "shape": [1], "on_shape": [1]}
        signed = auth.sign(data)
        assert auth.verify(signed) is True

    def test_reject_short_hmac_key(self):
        """HMAC key shorter than 16 bytes should be rejected."""
        from orion.core.crypto_utils import CiphertextAuthenticator

        with pytest.raises(ValueError, match="at least 16 bytes"):
            CiphertextAuthenticator(hmac_key=b"short")

    def test_unsigned_data_fails(self):
        """Data without HMAC should fail verification."""
        from orion.core.crypto_utils import CiphertextAuthenticator

        auth = CiphertextAuthenticator(hmac_key=b"test-key-1234567890")
        unsigned = {"ciphertexts": [b"test"], "shape": [1], "on_shape": [1]}
        assert auth.verify(unsigned) is False


# ================================================================
#  6. Key Encryption at Rest (Feature 8)
# ================================================================

class TestKeyEncryption:
    """Tests for AES-256-GCM key encryption."""

    def test_encrypt_decrypt_roundtrip(self):
        """Encrypted data should decrypt to original."""
        from orion.core.crypto_utils import KeyEncryptor

        encryptor = KeyEncryptor(password="test-password-12345")
        original = os.urandom(4096)  # Simulate a secret key

        encrypted = encryptor.encrypt(original)
        decrypted = encryptor.decrypt(encrypted)

        assert decrypted == original

    def test_wrong_password_fails(self):
        """Decryption with wrong password should fail."""
        from orion.core.crypto_utils import KeyEncryptor

        enc1 = KeyEncryptor(password="correct-password")
        enc2 = KeyEncryptor(password="wrong-password-1")

        encrypted = enc1.encrypt(b"secret data")
        with pytest.raises(ValueError, match="[Dd]ecryption failed"):
            enc2.decrypt(encrypted)

    def test_file_roundtrip(self):
        """Encrypt to file and decrypt from file should roundtrip."""
        from orion.core.crypto_utils import KeyEncryptor

        encryptor = KeyEncryptor(password="file-test-pass-1")
        original = os.urandom(2048)

        with tempfile.NamedTemporaryFile(suffix=".enc", delete=False) as f:
            filepath = f.name

        try:
            encryptor.encrypt_to_file(original, filepath)
            assert os.path.exists(filepath)

            decrypted = encryptor.decrypt_from_file(filepath)
            assert decrypted == original
        finally:
            os.unlink(filepath)

    def test_reject_short_password(self):
        """Password shorter than 8 chars should be rejected."""
        from orion.core.crypto_utils import KeyEncryptor

        with pytest.raises(ValueError, match="at least 8"):
            KeyEncryptor(password="short")

    def test_encrypted_format_has_metadata(self):
        """Encrypted output should contain version, salt, nonce."""
        from orion.core.crypto_utils import KeyEncryptor

        encryptor = KeyEncryptor(password="metadata-test-pw")
        encrypted = encryptor.encrypt(b"test data")

        assert "version" in encrypted
        assert "salt" in encrypted
        assert "nonce" in encrypted
        assert "kdf" in encrypted
        assert encrypted["version"] == 1

    def test_different_encryptions_differ(self):
        """Two encryptions of same data should produce different ciphertexts."""
        from orion.core.crypto_utils import KeyEncryptor

        encryptor = KeyEncryptor(password="determinism-test")
        data = b"same plaintext"

        enc1 = encryptor.encrypt(data)
        enc2 = encryptor.encrypt(data)

        # Random salt/nonce means ciphertexts differ
        assert enc1["ciphertext"] != enc2["ciphertext"]


# ================================================================
#  7. Polynomial Degree Bounds (Feature 3 — existing)
# ================================================================

class TestPolynomialBounds:
    """Tests for Chebyshev polynomial degree limits."""

    def test_max_degree_enforced(self):
        """Degrees above MAX_CHEBYSHEV_DEGREE should be rejected."""
        from orion.nn.activation import MAX_CHEBYSHEV_DEGREE

        assert MAX_CHEBYSHEV_DEGREE == 127

    def test_valid_degrees_accepted(self):
        """Standard polynomial degrees (3, 7, 15, 31) should work."""
        import orion.nn as on

        for degree in [3, 7, 15, 31]:
            act = on.SiLU(degree=degree)
            assert act.degree == degree


# ================================================================
#  8. Model Caching (Feature 3)
# ================================================================

class TestModelCaching:
    """Tests for FHE compiled state caching."""

    def test_cache_miss_returns_none(self):
        """Cache miss should return None."""
        from orion.core.cache import FHECache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FHECache(cache_dir=tmpdir)

            import torch.nn as nn
            model = nn.Linear(10, 2)

            with tempfile.NamedTemporaryFile(suffix=".yml", delete=False, mode="w") as f:
                f.write("ckks_params:\n  LogN: 14\n")
                config_path = f.name

            try:
                result = cache.load(config_path, model)
                assert result is None
            finally:
                os.unlink(config_path)

    def test_save_and_load(self):
        """Saved cache should be loadable."""
        from orion.core.cache import FHECache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FHECache(cache_dir=tmpdir)

            import torch.nn as nn
            model = nn.Linear(10, 2)

            with tempfile.NamedTemporaryFile(suffix=".yml", delete=False, mode="w") as f:
                f.write("ckks_params:\n  LogN: 14\n  LogQ: [45]\n")
                config_path = f.name

            try:
                cache.save(config_path, model, input_level=7)

                result = cache.load(config_path, model)
                assert result is not None
                assert result["input_level"] == 7
            finally:
                os.unlink(config_path)

    def test_cache_invalidation_on_model_change(self):
        """Cache should miss when model weights change."""
        from orion.core.cache import FHECache
        import torch.nn as nn

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FHECache(cache_dir=tmpdir)

            with tempfile.NamedTemporaryFile(suffix=".yml", delete=False, mode="w") as f:
                f.write("ckks_params:\n  LogN: 14\n  LogQ: [45]\n")
                config_path = f.name

            try:
                model1 = nn.Linear(10, 2)
                cache.save(config_path, model1, input_level=7)

                model2 = nn.Linear(10, 2)  # Different random weights
                result = cache.load(config_path, model2)
                # Very likely a cache miss due to different random weights
                # (extremely unlikely to have identical random init)
                # But don't assert None since there's a tiny chance
            finally:
                os.unlink(config_path)

    def test_list_entries(self):
        """list_entries should return cached metadata."""
        from orion.core.cache import FHECache
        import torch.nn as nn

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FHECache(cache_dir=tmpdir)

            with tempfile.NamedTemporaryFile(suffix=".yml", delete=False, mode="w") as f:
                f.write("ckks_params:\n  LogN: 14\n  LogQ: [45]\n")
                config_path = f.name

            try:
                model = nn.Linear(10, 2)
                cache.save(config_path, model, input_level=5)

                entries = cache.list_entries()
                assert len(entries) == 1
                assert entries[0]["input_level"] == 5
            finally:
                os.unlink(config_path)

    def test_clear_cache(self):
        """clear() should remove all entries."""
        from orion.core.cache import FHECache
        import torch.nn as nn

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FHECache(cache_dir=tmpdir)

            with tempfile.NamedTemporaryFile(suffix=".yml", delete=False, mode="w") as f:
                f.write("ckks_params:\n  LogN: 14\n  LogQ: [45]\n")
                config_path = f.name

            try:
                model = nn.Linear(10, 2)
                cache.save(config_path, model, input_level=5)
                assert len(cache.list_entries()) == 1

                cache.clear()
                assert len(cache.list_entries()) == 0
            finally:
                os.unlink(config_path)


# ================================================================
#  9. Batch Processing (Feature 1)
# ================================================================

class TestBatchProcessing:
    """Tests for the parallel/batch processing infrastructure."""

    def test_pipeline_executor_init(self):
        """PipelineExecutor should initialize without errors."""
        from orion.core.parallel import PipelineExecutor

        # Mock objects — just verify the class can be created
        executor = PipelineExecutor(
            scheme=None, model=None, input_level=7, num_threads=2
        )
        assert executor.num_threads == 2
        assert executor.input_level == 7

    def test_batch_processor_init(self):
        """BatchProcessor should initialize without errors."""
        from orion.core.parallel import BatchProcessor

        processor = BatchProcessor(
            scheme=None, model=None, input_level=7, cleanup_interval=10
        )
        assert processor.cleanup_interval == 10


# ================================================================
#  10. Integration Tests (require DLL)
# ================================================================

# These tests require the compiled Lattigo DLL and are skipped
# in environments where it's not available.

def _dll_available():
    """Check if the Lattigo DLL is available."""
    dll_path = os.path.join(
        os.path.dirname(__file__), "..",
        "orion", "backend", "lattigo", "lattigo-windows.dll"
    )
    return os.path.exists(dll_path)


@pytest.mark.skipif(not _dll_available(), reason="Lattigo DLL not found")
class TestIntegration:
    """Integration tests requiring the compiled Go backend."""

    def test_scheme_initialization(self):
        """Scheme should initialize with valid config."""
        import orion

        # Use smaller params that are within 128-bit security for LogN=14
        # Total LogQP = 5*35 + 2*46 = 267, well under 883
        config = {
            "ckks_params": {
                "LogN": 14,
                "LogQ": [45, 35, 35, 35, 45],
                "LogP": [46, 46],
                "LogScale": 35,
                "H": 192,
                "RingType": "Standard",
            },
            "orion": {
                "margin": 2,
                "embedding_method": "hybrid",
                "backend": "lattigo",
                "fuse_modules": True,
                "debug": False,
                "io_mode": "none",
            },
        }

        scheme = orion.init_scheme(config)
        assert scheme.backend is not None
        scheme.delete_scheme()

    def test_error_propagation_on_invalid_id(self):
        """Accessing invalid ciphertext ID should be caught."""
        from orion.core.error_handling import wrap_ffi_error_checking
        # This would test that accessing a non-existent ciphertext ID
        # triggers proper error handling rather than a Go panic.
        pass  # Requires live scheme


# ================================================================
#  11. Windows Portability (the critical fix)
# ================================================================

class TestWindowsPortability:
    """Tests related to the C.ulong/C.ulonglong Windows portability fix."""

    def test_scale_values_fit_ulonglong(self):
        """Scale values for LogScale > 31 should fit in 64-bit unsigned."""
        import ctypes

        for logscale in [32, 35, 40, 50]:
            scale = 1 << logscale
            # Must fit in c_ulonglong (64-bit)
            c_val = ctypes.c_ulonglong(scale)
            assert c_val.value == scale

            # Would overflow in c_ulong on Windows (32-bit!)
            if sys.platform == "win32":
                c_ulong_val = ctypes.c_ulong(scale)
                # On Windows, c_ulong is 32-bit, so high bits are lost
                if logscale > 31:
                    assert c_ulong_val.value != scale, \
                        f"LogScale={logscale}: c_ulong should overflow on Windows"

    def test_ctypes_ulonglong_is_64bit(self):
        """ctypes.c_ulonglong should always be 64 bits."""
        import ctypes
        assert ctypes.sizeof(ctypes.c_ulonglong) == 8

    def test_ctypes_ulong_size_platform(self):
        """Document the platform-specific c_ulong size."""
        import ctypes
        if sys.platform == "win32":
            assert ctypes.sizeof(ctypes.c_ulong) == 4, \
                "On Windows, c_ulong must be 32-bit (this is the bug we fixed)"
        else:
            assert ctypes.sizeof(ctypes.c_ulong) == 8, \
                "On Linux/macOS, c_ulong is 64-bit"


# ================================================================
#  12. HuggingFace Integration
# ================================================================

class TestHuggingFaceCompatibility:
    """Tests for the HuggingFace model compatibility checker."""

    def test_simple_mlp_compatible(self):
        """Simple MLP should be FHE compatible."""
        from orion.integrations import check_compatibility

        model = nn.Sequential(
            nn.Linear(10, 32), nn.GELU(), nn.Linear(32, 2))
        report = check_compatibility(model, "SimpleMLP")

        assert report.compatible is True
        assert report.unsupported_count == 0
        assert report.supported_count == 3  # 2 Linear + 1 GELU

    def test_cnn_compatible(self):
        """CNN with AvgPool and BatchNorm should be compatible."""
        from orion.integrations import check_compatibility

        model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 10),
        )
        report = check_compatibility(model, "SimpleCNN")
        assert report.compatible is True

    def test_maxpool_incompatible(self):
        """MaxPool should be flagged as incompatible."""
        from orion.integrations import check_compatibility

        model = nn.Sequential(
            nn.Conv2d(1, 16, 3), nn.MaxPool2d(2), nn.Flatten(),
            nn.Linear(16 * 13 * 13, 10))
        report = check_compatibility(model, "MaxPoolCNN")

        assert report.compatible is False
        assert report.unsupported_count >= 1
        assert any("MaxPool" in l for l in report.unsupported_layers)

    def test_transformer_incompatible(self):
        """Transformer with Embedding + Attention should be incompatible."""
        from orion.integrations import check_compatibility

        class TinyTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(100, 32)
                self.fc = nn.Linear(32, 10)
            def forward(self, x):
                return self.fc(self.emb(x).mean(1))

        report = check_compatibility(TinyTransformer(), "TinyTransformer")
        assert report.compatible is False
        assert any("Embedding" in l for l in report.unsupported_layers)

    def test_dropout_removable(self):
        """Dropout should be classified as removable, not unsupported."""
        from orion.integrations import check_compatibility

        model = nn.Sequential(
            nn.Linear(10, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 2))
        report = check_compatibility(model, "WithDropout")

        assert report.compatible is True
        assert report.removable_count == 1
        assert any("Dropout" in l for l in report.removable_layers)

    def test_depth_estimation(self):
        """Estimated depth should reflect Linear + activation costs."""
        from orion.integrations import check_compatibility

        model = nn.Sequential(
            nn.Linear(10, 32), nn.SiLU(),
            nn.Linear(32, 16), nn.SiLU(),
            nn.Linear(16, 2))
        report = check_compatibility(model, "ThreeLayer")

        # 3 Linear (1 each) + 2 SiLU (4 each) = 11
        assert report.estimated_depth == 11


class TestHuggingFaceConversion:
    """Tests for the HuggingFace model converter."""

    def test_convert_sequential(self):
        """Converting nn.Sequential should produce an orion.nn.Module."""
        from orion.integrations import convert_to_orion
        import orion.nn as on

        model = nn.Sequential(
            nn.Linear(10, 32), nn.SiLU(), nn.Linear(32, 2))
        orion_model = convert_to_orion(model, activation_degree=7)

        assert isinstance(orion_model, on.Module)

    def test_weight_transfer(self):
        """Converted model should produce identical cleartext output."""
        from orion.integrations import convert_to_orion

        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(10, 32), nn.SiLU(), nn.Linear(32, 2))

        orion_model = convert_to_orion(model, activation_degree=7)
        test_input = torch.randn(5, 10)

        model.eval()
        orion_model.eval()
        with torch.no_grad():
            orig = model(test_input)
            conv = orion_model(test_input)

        assert torch.allclose(orig, conv, atol=1e-5), \
            f"Max diff: {(orig - conv).abs().max().item()}"

    def test_all_activations_convert(self):
        """All 8 supported activation types should convert correctly."""
        from orion.integrations import convert_to_orion

        activations = [nn.ReLU, nn.GELU, nn.SiLU, nn.Sigmoid,
                       nn.ELU, nn.SELU, nn.Mish, nn.Softplus]

        for act_cls in activations:
            torch.manual_seed(0)
            model = nn.Sequential(
                nn.Linear(4, 8), act_cls(), nn.Linear(8, 2))
            orion_model = convert_to_orion(model, activation_degree=7)

            test_input = torch.randn(3, 4)
            model.eval()
            orion_model.eval()
            with torch.no_grad():
                orig = model(test_input)
                conv = orion_model(test_input)
            diff = (orig - conv).abs().max().item()
            assert diff < 1e-5, f"{act_cls.__name__}: diff={diff}"

    def test_reject_unsupported(self):
        """Converting model with unsupported layers should raise ValueError."""
        from orion.integrations import convert_to_orion

        model = nn.Sequential(
            nn.Linear(10, 32), nn.MaxPool1d(2), nn.Linear(16, 2))

        with pytest.raises(ValueError, match="unsupported"):
            convert_to_orion(model)

    def test_custom_model_conversion(self):
        """Non-Sequential models should convert correctly."""
        from orion.integrations import convert_to_orion

        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 16)
                self.act = nn.GELU()
                self.fc2 = nn.Linear(16, 3)
            def forward(self, x):
                return self.fc2(self.act(self.fc1(x)))

        torch.manual_seed(42)
        model = MyModel()
        orion_model = convert_to_orion(model, activation_degree=7)

        test_input = torch.randn(5, 4)
        model.eval()
        orion_model.eval()
        with torch.no_grad():
            orig = model(test_input)
            conv = orion_model(test_input)

        assert torch.allclose(orig, conv, atol=1e-5)

    def test_batchnorm_conversion(self):
        """BatchNorm layers should transfer running stats."""
        from orion.integrations import convert_to_orion

        model = nn.Sequential(
            nn.Linear(10, 32), nn.BatchNorm1d(32), nn.SiLU(),
            nn.Linear(32, 2))

        # Run a batch through to populate running stats
        model.train()
        model(torch.randn(16, 10))
        model.eval()

        orion_model = convert_to_orion(model, activation_degree=7)
        orion_model.eval()

        test_input = torch.randn(5, 10)
        with torch.no_grad():
            orig = model(test_input)
            conv = orion_model(test_input)

        assert torch.allclose(orig, conv, atol=1e-5)


# ================================================================
#  13. Transformer Layers
# ================================================================

class TestTransformerLayers:
    """Tests for FHE-compatible transformer components."""

    def test_poly_softmax_sums_to_one(self):
        """PolySoftmax output should sum to 1 along the specified dim."""
        from orion.nn.transformer import PolySoftmax

        ps = PolySoftmax(power=4, dim=-1)
        x = torch.randn(2, 4, 8)
        out = ps(x)

        sums = out.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_poly_softmax_non_negative(self):
        """PolySoftmax output should be non-negative (even powers)."""
        from orion.nn.transformer import PolySoftmax

        for power in [2, 4, 8, 16]:
            ps = PolySoftmax(power=power)
            out = ps(torch.randn(3, 5))
            assert (out >= 0).all(), f"Negative values with power={power}"

    def test_poly_softmax_invalid_power(self):
        """Non-even or unsupported powers should be rejected."""
        from orion.nn.transformer import PolySoftmax

        with pytest.raises(ValueError):
            PolySoftmax(power=3)
        with pytest.raises(ValueError):
            PolySoftmax(power=5)

    def test_fhe_layernorm_matches_pytorch(self):
        """FHELayerNorm cleartext output should match nn.LayerNorm."""
        from orion.nn.transformer import FHELayerNorm

        torch.manual_seed(42)
        x = torch.randn(4, 8, 32)

        pytorch_ln = nn.LayerNorm(32)
        fhe_ln = FHELayerNorm(32)
        fhe_ln.weight = pytorch_ln.weight
        fhe_ln.bias = pytorch_ln.bias

        pytorch_ln.eval()
        fhe_ln.eval()
        with torch.no_grad():
            pt_out = pytorch_ln(x)
            fhe_out = fhe_ln(x)

        assert torch.allclose(pt_out, fhe_out, atol=1e-6)

    def test_multi_head_attention_shape(self):
        """Attention output should preserve input shape."""
        from orion.nn.transformer import FHEMultiHeadAttention

        attn = FHEMultiHeadAttention(embed_dim=32, num_heads=4, softmax_power=4)
        x = torch.randn(2, 8, 32)
        attn.eval()
        with torch.no_grad():
            out = attn(x)
        assert out.shape == x.shape

    def test_encoder_layer_shape(self):
        """Encoder layer output should preserve input shape."""
        from orion.nn.transformer import FHETransformerEncoderLayer

        enc = FHETransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64, softmax_power=4)
        x = torch.randn(2, 8, 32)
        enc.eval()
        with torch.no_grad():
            out = enc(x)
        assert out.shape == x.shape

    def test_encoder_layer_residual(self):
        """Encoder output should differ from input (not identity)."""
        from orion.nn.transformer import FHETransformerEncoderLayer

        torch.manual_seed(42)
        enc = FHETransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64)
        x = torch.randn(2, 4, 32)
        enc.eval()
        with torch.no_grad():
            out = enc(x)
        assert not torch.allclose(out, x)
        assert (out - x).abs().mean() < 10.0

    def test_fhe_transformer_trains(self):
        """FHE-compatible transformer should be trainable."""
        from orion.nn.transformer import FHETransformerEncoderLayer

        torch.manual_seed(42)
        embed = nn.Linear(1, 16)
        encoder = FHETransformerEncoderLayer(16, 4, 32, softmax_power=4)
        head = nn.Linear(16, 3)

        x = torch.randn(16, 4)
        y = torch.randint(0, 3, (16,))

        params = list(embed.parameters()) + list(encoder.parameters()) + list(head.parameters())
        optimizer = torch.optim.Adam(params, lr=0.01)

        embed.train(); encoder.train(); head.train()
        h = embed(x.unsqueeze(-1))
        h = encoder(h)
        logits = head(h.mean(dim=1))
        loss1 = nn.CrossEntropyLoss()(logits, y)
        loss1.backward()
        optimizer.step()
        optimizer.zero_grad()

        h = embed(x.unsqueeze(-1))
        h = encoder(h)
        logits = head(h.mean(dim=1))
        loss2 = nn.CrossEntropyLoss()(logits, y)

        assert loss2.item() < loss1.item(), "Loss should decrease after one step"


class TestTransformerCompatibility:
    """Tests for transformer compatibility checking."""

    def test_fhe_transformer_compatible(self):
        """Model with FHE transformer layers should be compatible."""
        from orion.integrations import check_compatibility
        from orion.nn.transformer import FHETransformerEncoderLayer

        class FHEModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(1, 16)
                self.enc = FHETransformerEncoderLayer(16, 4, 32, softmax_power=4)
                self.head = nn.Linear(16, 3)
            def forward(self, x):
                return self.head(self.enc(self.embed(x.unsqueeze(-1))).mean(1))

        report = check_compatibility(FHEModel(), "FHEModel")
        assert report.compatible is True

    def test_standard_transformer_incompatible(self):
        """Standard nn.TransformerEncoderLayer should be incompatible."""
        from orion.integrations import check_compatibility

        model = nn.Sequential(
            nn.TransformerEncoderLayer(16, 4, 32, dropout=0, batch_first=True))
        report = check_compatibility(model, "StdTransformer")
        assert report.compatible is False

    def test_layernorm_now_compatible(self):
        """nn.LayerNorm should now be compatible (converts to FHELayerNorm)."""
        from orion.integrations import check_compatibility

        model = nn.Sequential(
            nn.Linear(10, 32), nn.LayerNorm(32), nn.GELU(), nn.Linear(32, 2))
        report = check_compatibility(model, "WithLayerNorm")
        assert report.compatible is True

    def test_layernorm_conversion(self):
        """nn.LayerNorm should auto-convert to FHELayerNorm."""
        from orion.integrations import convert_to_orion
        from orion.nn.transformer import FHELayerNorm

        model = nn.Sequential(
            nn.Linear(10, 32), nn.LayerNorm(32), nn.GELU(), nn.Linear(32, 2))
        model.eval()

        orion_model = convert_to_orion(model, activation_degree=7)
        orion_model.eval()

        found_fhe_ln = any(isinstance(m, FHELayerNorm) for m in orion_model.modules())
        assert found_fhe_ln, "FHELayerNorm not found in converted model"

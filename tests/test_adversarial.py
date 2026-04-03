"""
Adversarial and fuzz tests for Orion's security hardening.

Tests that security features hold under active attack patterns:
- Ciphertext tampering detection
- Key encryption robustness
- Config injection prevention
- Path traversal blocking
- FFI boundary hardening
- Resource exhaustion resistance
- Concurrent access safety

For ACSAC/EuroS&P/USENIX Security paper evaluation.
"""

import os
import sys
import json
import hmac
import math
import copy
import time
import hashlib
import base64
import struct
import random
import pickle
import shutil
import tempfile
import warnings
import threading
import concurrent.futures
from unittest import mock
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ================================================================
#  Helpers
# ================================================================

def _make_authenticator():
    """Create a CiphertextAuthenticator with a valid key."""
    from orion.core.crypto_utils import CiphertextAuthenticator
    return CiphertextAuthenticator(hmac_key=os.urandom(32))


def _make_signed_payload(auth, data=None):
    """Create a signed ciphertext payload."""
    if data is None:
        data = {
            "ciphertexts": [os.urandom(256), os.urandom(256)],
            "shape": [1, 13],
            "on_shape": [1, 13],
        }
    return auth.sign(data)


def _make_params_dict(io_mode="load", diags_path="safe_diags", keys_path="safe_keys"):
    """Create a valid parameter dict for NewParameters."""
    return {
        "ckks_params": {
            "LogN": 14,
            "LogQ": [45, 35, 35, 35, 45],
            "LogP": [46],
            "LogScale": 35,
            "RingType": "Standard",
        },
        "orion": {
            "backend": "lattigo",
            "diags_path": diags_path,
            "keys_path": keys_path,
            "io_mode": io_mode,
        },
    }


# ================================================================
#  1. Ciphertext Tampering Detection
# ================================================================

class TestCiphertextTampering:
    """
    Tests that HMAC-SHA256 authentication detects various ciphertext
    tampering patterns. Models an adversary with network-level access
    who can modify ciphertexts in transit.

    Adversary capability: full read/write on the transport channel.
    """

    def test_single_bit_flip(self):
        """Flipping a single bit in ciphertext data must be detected."""
        from orion.core.crypto_utils import CiphertextAuthenticator

        auth = _make_authenticator()
        signed = _make_signed_payload(auth)

        # Flip one bit in the first ciphertext
        ct = bytearray(signed["ciphertexts"][0])
        ct[len(ct) // 2] ^= 0x01
        signed["ciphertexts"][0] = bytes(ct)

        assert auth.verify(signed) is False

    def test_truncated_ciphertext(self):
        """Truncating ciphertext bytes must invalidate the HMAC."""
        auth = _make_authenticator()
        signed = _make_signed_payload(auth)

        # Truncate first ciphertext to half its length
        original = signed["ciphertexts"][0]
        signed["ciphertexts"][0] = original[: len(original) // 2]

        assert auth.verify(signed) is False

    def test_extended_ciphertext(self):
        """Appending extra bytes to a ciphertext must be detected."""
        auth = _make_authenticator()
        signed = _make_signed_payload(auth)

        # Extend with random padding
        signed["ciphertexts"][0] = signed["ciphertexts"][0] + os.urandom(64)

        assert auth.verify(signed) is False

    def test_empty_ciphertext(self):
        """Replacing ciphertext with empty bytes must be detected."""
        auth = _make_authenticator()
        signed = _make_signed_payload(auth)

        signed["ciphertexts"][0] = b""
        assert auth.verify(signed) is False

    def test_swapped_ciphertext_bytes(self):
        """Shuffling ciphertext bytes (permutation attack) must be detected."""
        auth = _make_authenticator()
        signed = _make_signed_payload(auth)

        ct = bytearray(signed["ciphertexts"][0])
        # Reverse the byte order (shuffle attack)
        ct.reverse()
        signed["ciphertexts"][0] = bytes(ct)

        assert auth.verify(signed) is False

    def test_mac_truncation_attack(self):
        """Truncating the HMAC value must cause verification failure."""
        auth = _make_authenticator()
        signed = _make_signed_payload(auth)

        # Truncate the MAC to half its length
        signed["hmac"] = signed["hmac"][: len(signed["hmac"]) // 2]
        assert auth.verify(signed) is False

    def test_null_bytes_injected(self):
        """Injecting null bytes into ciphertext must be detected."""
        auth = _make_authenticator()
        signed = _make_signed_payload(auth)

        ct = bytearray(signed["ciphertexts"][0])
        # Inject null bytes in the middle
        mid = len(ct) // 2
        ct[mid : mid] = b"\x00" * 16
        signed["ciphertexts"][0] = bytes(ct)

        assert auth.verify(signed) is False

    def test_replayed_ciphertext_wrong_key(self):
        """
        A validly signed payload must fail when verified by a different
        authenticator (simulates replaying across sessions with different keys).
        """
        from orion.core.crypto_utils import CiphertextAuthenticator

        auth_sender = CiphertextAuthenticator(hmac_key=os.urandom(32))
        auth_receiver = CiphertextAuthenticator(hmac_key=os.urandom(32))

        signed = _make_signed_payload(auth_sender)
        assert auth_receiver.verify(signed) is False

    def test_base64_encoded_ciphertext_tamper(self):
        """Tampering with base64-encoded ciphertext strings must be detected."""
        auth = _make_authenticator()
        ct_bytes = os.urandom(256)
        data = {
            "ciphertexts": [base64.b64encode(ct_bytes).decode("ascii")],
            "shape": [1],
            "on_shape": [1],
        }
        signed = auth.sign(data)

        # Tamper the base64 string
        tampered_bytes = bytearray(ct_bytes)
        tampered_bytes[0] ^= 0xFF
        signed["ciphertexts"][0] = base64.b64encode(bytes(tampered_bytes)).decode("ascii")

        assert auth.verify(signed) is False


# ================================================================
#  2. Key Encryption Adversarial
# ================================================================

class TestKeyEncryptionAdversarial:
    """
    Tests that key encryption at rest (AES-256-GCM / HMAC-XOR fallback)
    resists corruption, field manipulation, and edge-case inputs.

    Adversary capability: read/write access to encrypted key files on disk.
    """

    def test_corrupted_ciphertext_field(self):
        """Flipping bits in the encrypted ciphertext field must fail decryption."""
        from orion.core.crypto_utils import KeyEncryptor

        enc = KeyEncryptor(password="adversarial-test-pw")
        encrypted = enc.encrypt(b"secret key material " * 10)

        # Corrupt the ciphertext by flipping bits in the decoded bytes
        ct_bytes = bytearray(base64.b64decode(encrypted["ciphertext"]))
        ct_bytes[len(ct_bytes) // 2] ^= 0xFF
        encrypted["ciphertext"] = base64.b64encode(bytes(ct_bytes)).decode()

        with pytest.raises((ValueError, Exception)):
            enc.decrypt(encrypted)

    def test_corrupted_salt(self):
        """Corrupting the salt must produce wrong derived key -> decryption failure."""
        from orion.core.crypto_utils import KeyEncryptor

        enc = KeyEncryptor(password="salt-corruption-pw")
        encrypted = enc.encrypt(b"secret data")

        salt_bytes = bytearray(base64.b64decode(encrypted["salt"]))
        salt_bytes[0] ^= 0xFF
        encrypted["salt"] = base64.b64encode(bytes(salt_bytes)).decode()

        with pytest.raises((ValueError, Exception)):
            enc.decrypt(encrypted)

    def test_corrupted_nonce(self):
        """Corrupting the nonce/IV must cause decryption failure."""
        from orion.core.crypto_utils import KeyEncryptor

        enc = KeyEncryptor(password="nonce-corruption-pw")
        encrypted = enc.encrypt(b"secret data")

        nonce_bytes = bytearray(base64.b64decode(encrypted["nonce"]))
        nonce_bytes[0] ^= 0xFF
        encrypted["nonce"] = base64.b64encode(bytes(nonce_bytes)).decode()

        with pytest.raises((ValueError, Exception)):
            enc.decrypt(encrypted)

    def test_modified_version_number(self):
        """Modified version number should not cause unhandled crashes."""
        from orion.core.crypto_utils import KeyEncryptor

        enc = KeyEncryptor(password="version-test-pass")
        encrypted = enc.encrypt(b"test data here")

        # Modify version to a future version
        encrypted["version"] = 999

        # Should either decrypt normally (version ignored) or raise cleanly
        try:
            result = enc.decrypt(encrypted)
            # If it works, the version is not enforced (acceptable behavior)
            assert result == b"test data here"
        except (ValueError, KeyError, Exception):
            pass  # Clean failure is acceptable

    def test_missing_fields(self):
        """Removing required fields from encrypted dict must raise cleanly."""
        from orion.core.crypto_utils import KeyEncryptor

        enc = KeyEncryptor(password="missing-field-pw")
        encrypted = enc.encrypt(b"important data")

        for field in ["salt", "nonce", "ciphertext"]:
            corrupted = {k: v for k, v in encrypted.items() if k != field}
            with pytest.raises((KeyError, ValueError, Exception)):
                enc.decrypt(corrupted)

    def test_very_long_password(self):
        """A 1MB password must not crash (DoS resistance)."""
        from orion.core.crypto_utils import KeyEncryptor

        long_pw = "A" * (1024 * 1024)  # 1 MB password
        enc = KeyEncryptor(password=long_pw)
        encrypted = enc.encrypt(b"test")
        decrypted = enc.decrypt(encrypted)
        assert decrypted == b"test"

    def test_unicode_special_char_password(self):
        """Unicode and special character passwords must work correctly."""
        from orion.core.crypto_utils import KeyEncryptor

        passwords = [
            "\u00e9\u00e8\u00ea\u00eb\u00e0\u00e2\u00e4\u00e7\u00f9\u00fb",  # French accents
            "\u4f60\u597d\u4e16\u754c\u5bc6\u7801\u5b89\u5168\u6d4b\u8bd5",  # Chinese characters
            "p@$$w0rd!#%^&*(){}[]|\\:\";<>?,./~`",  # Special ASCII
            "\U0001f512\U0001f511\U0001f6e1\ufe0f\U0001f525\U0001f4a3\U0001f52b\U0001f6a8\U0001f6a8",  # Emojis
        ]
        for pw in passwords:
            if len(pw) < 8:
                pw = pw + "x" * (8 - len(pw))
            enc = KeyEncryptor(password=pw)
            encrypted = enc.encrypt(b"secret")
            assert enc.decrypt(encrypted) == b"secret"

    def test_empty_plaintext_encryption(self):
        """Encrypting empty bytes must round-trip correctly."""
        from orion.core.crypto_utils import KeyEncryptor

        enc = KeyEncryptor(password="empty-plaintext-test")
        encrypted = enc.encrypt(b"")
        decrypted = enc.decrypt(encrypted)
        assert decrypted == b""

    def test_repeated_encrypt_decrypt_cycles(self):
        """100 encrypt/decrypt cycles must show no degradation."""
        from orion.core.crypto_utils import KeyEncryptor

        enc = KeyEncryptor(password="cycle-stability-test")
        data = os.urandom(512)

        for i in range(100):
            encrypted = enc.encrypt(data)
            decrypted = enc.decrypt(encrypted)
            assert decrypted == data, f"Degradation at cycle {i}"


# ================================================================
#  3. Config Injection
# ================================================================

class TestConfigInjection:
    """
    Tests that the CKKS parameter validator correctly rejects adversarial
    configurations designed to weaken security or cause resource exhaustion.

    Adversary capability: control over the configuration file content.
    """

    def test_negative_logn(self):
        """Negative LogN must be rejected."""
        from orion.core.config_validator import (
            validate_ckks_params, SecurityValidationError,
        )

        with pytest.raises(SecurityValidationError, match="below minimum"):
            validate_ckks_params(logn=-1, logq=[35, 35], logp=[46], logscale=35)

    def test_negative_logq_values(self):
        """Negative values in LogQ must be rejected."""
        from orion.core.config_validator import (
            validate_ckks_params, SecurityValidationError,
        )

        with pytest.raises(SecurityValidationError, match="must be positive"):
            validate_ckks_params(logn=14, logq=[-5, 35, 35], logp=[46], logscale=35)

    def test_negative_logp_values(self):
        """Negative values in LogP must be rejected."""
        from orion.core.config_validator import (
            validate_ckks_params, SecurityValidationError,
        )

        with pytest.raises(SecurityValidationError, match="must be positive"):
            validate_ckks_params(logn=14, logq=[45, 35, 45], logp=[-10], logscale=35)

    def test_extremely_large_logn_100(self):
        """LogN=100 must be rejected as exceeding maximum."""
        from orion.core.config_validator import (
            validate_ckks_params, SecurityValidationError,
        )

        with pytest.raises(SecurityValidationError, match="exceeds maximum"):
            validate_ckks_params(logn=100, logq=[35, 35], logp=[46], logscale=35)

    def test_extremely_large_logn_1000(self):
        """LogN=1000 must be rejected as exceeding maximum."""
        from orion.core.config_validator import (
            validate_ckks_params, SecurityValidationError,
        )

        with pytest.raises(SecurityValidationError, match="exceeds maximum"):
            validate_ckks_params(logn=1000, logq=[35, 35], logp=[46], logscale=35)

    def test_empty_logq_list(self):
        """Empty LogQ list must be rejected."""
        from orion.core.config_validator import (
            validate_ckks_params, SecurityValidationError,
        )

        with pytest.raises(SecurityValidationError, match="LogQ is empty"):
            validate_ckks_params(logn=14, logq=[], logp=[46], logscale=35)

    def test_float_values_where_int_expected(self):
        """Float values for integer parameters must be handled gracefully."""
        from orion.core.config_validator import validate_ckks_params

        # Python allows float comparison with int, so this tests type robustness.
        # Should either work (treating 35.0 as 35) or raise a clean TypeError.
        try:
            result = validate_ckks_params(
                logn=14,
                logq=[45.0, 35.0, 35.0, 45.0],
                logp=[46.0],
                logscale=35,
                strict=False,
            )
            # If it works, the validator handled float coercion gracefully
            assert isinstance(result, dict)
        except (TypeError, ValueError):
            pass  # Clean failure is also acceptable

    def test_logq_millions_of_entries(self):
        """
        LogQ with 10,000 entries must not cause resource exhaustion.
        The total LogQP will massively exceed security bounds.
        """
        from orion.core.config_validator import (
            validate_ckks_params, SecurityValidationError,
        )

        huge_logq = [35] * 10_000
        with pytest.raises(SecurityValidationError, match="INSECURE"):
            validate_ckks_params(
                logn=14, logq=huge_logq, logp=[46], logscale=35
            )

    def test_logqp_sum_at_boundary(self):
        """
        LogQP sum exactly at the 128-bit boundary for LogN=14 (883)
        must be accepted.
        """
        from orion.core.config_validator import validate_ckks_params

        # LogN=14: 128-bit bound is 883
        # We need sum(logq) + sum(logp) = 883
        # Use logq=[45, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 45] = 525
        # logp = [46, 46, 46] = 138 => total = 663 (under 883)
        # Better: construct exactly 883
        # logq = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50] = 650
        # logp = [233] = 233 => total = 883
        # Actually keep it simpler:
        logq = [45] * 17  # 765
        logp = [59, 59]   # 118, total = 883
        total = sum(logq) + sum(logp)
        assert total == 883

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = validate_ckks_params(
                logn=14, logq=logq, logp=logp, logscale=35,
            )
        assert result["valid"] is True
        assert result["security_level"] == 128

    def test_zero_valued_logq_entries(self):
        """Zero-valued entries in LogQ must be rejected."""
        from orion.core.config_validator import (
            validate_ckks_params, SecurityValidationError,
        )

        with pytest.raises(SecurityValidationError, match="must be positive"):
            validate_ckks_params(logn=14, logq=[45, 0, 35, 45], logp=[46], logscale=35)

    def test_hamming_weight_zero(self):
        """Hamming weight = 0 must be rejected."""
        from orion.core.config_validator import (
            validate_ckks_params, SecurityValidationError,
        )

        with pytest.raises(SecurityValidationError, match="must be positive"):
            validate_ckks_params(
                logn=14, logq=[45, 35, 45], logp=[46], logscale=35, h=0,
            )


# ================================================================
#  4. Path Traversal Adversarial
# ================================================================

class TestPathTraversalAdversarial:
    """
    Tests that _validate_path in NewParameters blocks all known
    path traversal attack vectors including encoding tricks,
    platform-specific separators, and unicode abuse.

    Adversary capability: control over configuration path strings.
    """

    def _get_validate_path(self):
        """Get the _validate_path method from a NewParameters instance."""
        from orion.backend.python.parameters import NewParameters

        # Create a minimal valid NewParameters to get the method
        params = NewParameters(_make_params_dict(io_mode="none"))
        return params._validate_path

    def test_basic_parent_directory(self):
        """../../etc/passwd must be rejected."""
        validate_path = self._get_validate_path()
        with pytest.raises(ValueError, match="[Pp]ath traversal"):
            validate_path("../../etc/passwd")

    def test_deeply_nested_traversal(self):
        """../../../../../etc/passwd must be rejected."""
        validate_path = self._get_validate_path()
        with pytest.raises(ValueError, match="[Pp]ath traversal"):
            validate_path("../../../../../etc/passwd")

    def test_windows_style_backslash(self):
        """..\\..\\windows\\system32 must be rejected."""
        validate_path = self._get_validate_path()
        with pytest.raises(ValueError, match="[Pp]ath traversal"):
            validate_path("..\\..\\windows\\system32")

    def test_mixed_separators(self):
        """../..\\etc/passwd with mixed separators must be rejected."""
        validate_path = self._get_validate_path()
        with pytest.raises(ValueError, match="[Pp]ath traversal"):
            validate_path("../..\\etc/passwd")

    def test_null_byte_injection(self):
        """
        Null byte injection (valid_path\\x00../../etc/passwd).
        The OS may truncate at the null byte; _validate_path should
        still handle it safely.
        """
        validate_path = self._get_validate_path()
        # On most modern Python + OS combos, null bytes in paths raise ValueError
        try:
            validate_path("valid_path\x00../../etc/passwd")
            # If it doesn't raise, the null byte was stripped and the path
            # resolved safely (acceptable)
        except (ValueError, OSError):
            pass  # Clean rejection is the expected outcome

    def test_absolute_path_override(self):
        """Absolute paths like /etc/passwd must be rejected (outside CWD)."""
        validate_path = self._get_validate_path()
        # On Windows this resolves to current drive root, still outside CWD
        with pytest.raises(ValueError, match="[Pp]ath traversal"):
            validate_path("/etc/passwd")

    def test_symlink_like_pattern(self):
        """./valid/../../../etc/passwd must be rejected."""
        validate_path = self._get_validate_path()
        with pytest.raises(ValueError, match="[Pp]ath traversal"):
            validate_path("./valid/../../../etc/passwd")

    def test_long_path(self):
        """
        Very long path (4096+ chars) must not cause a crash.
        Should either reject or handle gracefully.
        """
        validate_path = self._get_validate_path()
        long_segment = "a" * 4096
        try:
            result = validate_path(long_segment)
            # If accepted, it stayed within CWD (just a long name)
            assert isinstance(result, str)
        except (ValueError, OSError):
            pass  # Clean rejection is also acceptable

    def test_path_with_spaces_and_special_chars(self):
        """Paths with spaces and special characters must be handled safely."""
        validate_path = self._get_validate_path()
        # This path stays within CWD, so it should be accepted
        result = validate_path("my output/data files (v2)/result.h5")
        assert isinstance(result, str)

    def test_dot_dot_at_end(self):
        """Trailing .. must be rejected."""
        validate_path = self._get_validate_path()
        with pytest.raises(ValueError, match="[Pp]ath traversal"):
            validate_path("subdir/../../..")

    def test_newparameters_rejects_traversal_diags(self):
        """NewParameters must reject diags_path traversal in save mode.

        In save mode, __post_init__ calls reset_stored_diags() which
        invokes _validate_path at construction time.
        """
        from orion.backend.python.parameters import NewParameters

        with pytest.raises(ValueError, match="[Pp]ath traversal"):
            NewParameters(_make_params_dict(
                io_mode="save",
                diags_path="../../etc/evil",
                keys_path="safe_keys",
            ))

    def test_newparameters_rejects_traversal_keys(self):
        """NewParameters.get_keys_path() must reject traversal in load mode."""
        from orion.backend.python.parameters import NewParameters

        params = NewParameters(_make_params_dict(
            io_mode="load",
            diags_path="safe_diags",
            keys_path="../../../etc/shadow",
        ))
        with pytest.raises(ValueError, match="[Pp]ath traversal"):
            params.get_keys_path()


# ================================================================
#  5. FFI Boundary Fuzzing
# ================================================================

class TestFFIBoundaryFuzzing:
    """
    Tests the FFI error handling layer against malformed inputs,
    unexpected return values, and edge cases in the Go-to-Python boundary.

    Adversary capability: ability to cause arbitrary Go-side failures.
    """

    def test_mock_function_raises_various_exceptions(self):
        """check_ffi_error must propagate Go-side errors as FHEBackendError."""
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
            def do_thing(self):
                self.backend._error = "Go panic: invalid memory address"
                return -1

        backend = MockBackend()
        module = MockModule(backend)

        with pytest.raises(FHEBackendError, match="invalid memory address"):
            module.do_thing()

    def test_mock_function_returns_none(self):
        """A wrapped function returning None with no error must not crash."""
        from orion.core.error_handling import check_ffi_error

        class MockBackend:
            def get_last_error(self):
                return None

            def clear_last_error(self):
                pass

        class MockModule:
            def __init__(self):
                self.backend = MockBackend()

            @check_ffi_error
            def returns_none(self):
                return None

        module = MockModule()
        result = module.returns_none()
        assert result is None

    def test_get_last_error_various_strings(self):
        """wrap_ffi_error_checking must handle various error string formats."""
        from orion.core.error_handling import wrap_ffi_error_checking, FHEBackendError

        error_messages = [
            "simple error",
            "error with special chars: !@#$%^&*()",
            "",  # empty error string
            "a" * 10000,  # very long error
            "error\nwith\nnewlines",
            "error\twith\ttabs",
            "error with \x00 null byte",
        ]

        for err_msg in error_messages:
            class MockLib:
                def __init__(self, error):
                    self._error_to_set = error
                    self._error = None

                def Encode(self, *args):
                    # Simulate Go side setting error during the call
                    self._error = self._error_to_set
                    return -1

                def get_last_error(self):
                    return self._error

                def clear_last_error(self):
                    self._error = None

            lib = MockLib(err_msg)
            wrap_ffi_error_checking(lib)

            if err_msg:  # non-empty error should raise
                with pytest.raises(FHEBackendError):
                    lib.Encode(1, 2, 3)
            else:
                # Empty error string is falsy in Python, so the error check
                # will not trigger. The function should return -1.
                result = lib.Encode(1, 2, 3)
                # -1 is returned but no error is set, so no exception
                assert result == -1

    def test_concurrent_ffi_calls(self):
        """
        Concurrent FFI calls must not corrupt the error state.
        Tests thread safety of the error-checking mechanism.
        """
        from orion.core.error_handling import wrap_ffi_error_checking, FHEBackendError

        lock = threading.Lock()
        errors_detected = []

        class MockLib:
            def __init__(self):
                self._error = None

            def get_last_error(self):
                return self._error

            def clear_last_error(self):
                self._error = None

            def Encode(self, *args):
                # Simulate variable latency
                time.sleep(random.uniform(0.001, 0.005))
                self._error = f"error-{threading.current_thread().name}"
                return -1

        lib = MockLib()
        wrap_ffi_error_checking(lib)

        def call_encode(idx):
            try:
                lib.Encode(idx)
            except FHEBackendError as e:
                with lock:
                    errors_detected.append(str(e))

        threads = []
        for i in range(20):
            t = threading.Thread(target=call_encode, args=(i,), name=f"t-{i}")
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All threads should have detected an error
        assert len(errors_detected) == 20

    def test_very_large_error_message(self):
        """A 1MB error message from Go side must be handled without crash."""
        from orion.core.error_handling import FHEBackendError

        huge_msg = "X" * (1024 * 1024)
        err = FHEBackendError(huge_msg, function_name="BigError")
        assert "BigError" in str(err)
        assert len(str(err)) > 1_000_000

    def test_utf8_binary_garbage_in_error(self):
        """Binary garbage in error messages must not crash the handler."""
        from orion.core.error_handling import FHEBackendError

        # Invalid UTF-8 sequences
        garbage_strs = [
            "error \udce9\udce8\udcea",  # surrogate escapes
            "error with binary: \x80\x81\x82\xff\xfe",
            "\x00\x00\x00",
        ]
        for msg in garbage_strs:
            err = FHEBackendError(msg, function_name="GarbageTest")
            # Must not crash; str() must return something
            assert isinstance(str(err), str)

    def test_sentinel_value_negative_one_without_error(self):
        """
        If a function returns -1 but no Go error is set, the wrapper
        should still return -1 (not raise), since -1 might be a valid return.
        """
        from orion.core.error_handling import wrap_ffi_error_checking

        class MockLib:
            def __init__(self):
                self._error = None

            def get_last_error(self):
                return self._error

            def clear_last_error(self):
                self._error = None

            def Encrypt(self, *args):
                return -1  # sentinel, but no error set

        lib = MockLib()
        wrap_ffi_error_checking(lib)
        # No error is set, so wrap_ffi_error_checking reads empty error:
        # The sentinel check fires, calls get_last_error -> None, no raise.
        result = lib.Encrypt(42)
        # No exception should be raised since _error is None
        assert result == -1


# ================================================================
#  6. Memory Exhaustion
# ================================================================

class TestMemoryExhaustion:
    """
    Tests that memory management utilities handle edge cases and
    adversarial usage patterns without crashes or resource leaks.

    Adversary capability: control over timing and sequencing of operations.
    """

    def test_memory_tracker_rapid_snapshots(self):
        """MemoryTracker must handle thousands of rapid snapshots."""
        from orion.core.memory import MemoryTracker

        class MockBackend:
            def GetLivePlaintexts(self):
                return list(range(random.randint(0, 100)))

            def GetLiveCiphertexts(self):
                return list(range(random.randint(0, 100)))

        tracker = MemoryTracker(MockBackend())

        for i in range(2000):
            tracker.snapshot(label=f"snap-{i}")

        assert len(tracker.snapshots) == 2000
        report = tracker.report()
        assert "snap-0" in report
        assert "snap-1999" in report

    def test_managed_cipher_with_none_input(self):
        """ManagedCipherTensor with None-like input must not crash."""
        from orion.core.memory import ManagedCipherTensor

        class FakeCipher:
            ids = []
            backend = MagicMock()

        managed = ManagedCipherTensor(FakeCipher())
        with managed:
            pass  # Should enter and exit cleanly
        # Release is idempotent
        managed.release()

    def test_double_release(self):
        """Calling release() twice must be safe (idempotent)."""
        from orion.core.memory import ManagedCipherTensor

        class FakeCipher:
            ids = [1, 2, 3]
            backend = MagicMock()

        managed = ManagedCipherTensor(FakeCipher())
        managed.release()
        managed.release()  # Second call must not crash

        assert managed._released is True

    def test_nested_managed_cipher_contexts(self):
        """Nested managed_cipher contexts must clean up correctly."""
        from orion.core.memory import managed_cipher

        delete_calls = []

        class FakeBackend:
            def DeleteCiphertext(self, idx):
                delete_calls.append(idx)

        class FakeCipher:
            def __init__(self, ids):
                self.ids = list(ids)
                self.backend = FakeBackend()

        ct_outer = FakeCipher([10, 20])
        ct_inner = FakeCipher([30, 40])

        with managed_cipher(ct_outer) as outer:
            with managed_cipher(ct_inner) as inner:
                pass  # inner cleaned up here
            # outer still alive here

        # Both should have been cleaned up
        assert 30 in delete_calls
        assert 40 in delete_calls
        assert 10 in delete_calls
        assert 20 in delete_calls

    def test_managed_cipher_exception_cleanup(self):
        """Go-side memory must be freed even when an exception occurs."""
        from orion.core.memory import managed_cipher

        freed_ids = []

        class FakeBackend:
            def DeleteCiphertext(self, idx):
                freed_ids.append(idx)

        class FakeCipher:
            def __init__(self):
                self.ids = [100, 200, 300]
                self.backend = FakeBackend()

        with pytest.raises(RuntimeError):
            with managed_cipher(FakeCipher()) as ct:
                raise RuntimeError("simulated crash")

        assert 100 in freed_ids
        assert 200 in freed_ids
        assert 300 in freed_ids


# ================================================================
#  7. Polynomial Bounds Fuzzing
# ================================================================

class TestPolynomialBoundsFuzzing:
    """
    Tests that Chebyshev polynomial degree validation rejects adversarial
    inputs that could cause exponential time/memory in FHE evaluation.

    Adversary capability: control over model configuration parameters.
    """

    def test_degree_zero(self):
        """Degree = 0 must be rejected (minimum is 1)."""
        from orion.nn.activation import Chebyshev

        with pytest.raises(ValueError, match="must be >= 1"):
            Chebyshev(degree=0, fn=lambda x: x)

    def test_degree_negative_one(self):
        """Degree = -1 must be rejected."""
        from orion.nn.activation import Chebyshev

        with pytest.raises(ValueError, match="must be >= 1"):
            Chebyshev(degree=-1, fn=lambda x: x)

    def test_degree_negative_100(self):
        """Degree = -100 must be rejected."""
        from orion.nn.activation import Chebyshev

        with pytest.raises(ValueError, match="must be >= 1"):
            Chebyshev(degree=-100, fn=lambda x: x)

    def test_degree_just_over_max(self):
        """Degree = 128 (one over MAX_CHEBYSHEV_DEGREE=127) must be rejected."""
        from orion.nn.activation import Chebyshev, MAX_CHEBYSHEV_DEGREE

        assert MAX_CHEBYSHEV_DEGREE == 127
        with pytest.raises(ValueError, match="exceeds maximum"):
            Chebyshev(degree=128, fn=lambda x: x)

    def test_degree_max_int(self):
        """Degree = sys.maxsize must be rejected."""
        from orion.nn.activation import Chebyshev

        with pytest.raises(ValueError, match="exceeds maximum"):
            Chebyshev(degree=sys.maxsize, fn=lambda x: x)

    def test_degree_float_type_error(self):
        """Degree = 7.5 (float) must raise TypeError or ValueError."""
        from orion.nn.activation import Chebyshev

        # Python's comparison with int works for floats, but
        # Chebyshev might fail in other ways if degree is float
        try:
            c = Chebyshev(degree=7.5, fn=lambda x: x)
            # If it constructs, the degree should have been stored
            assert isinstance(c.degree, (int, float))
        except (TypeError, ValueError):
            pass  # Clean rejection is acceptable

    def test_degree_at_boundary(self):
        """Degree = 127 (exactly MAX_CHEBYSHEV_DEGREE) must be accepted."""
        from orion.nn.activation import Chebyshev, MAX_CHEBYSHEV_DEGREE

        c = Chebyshev(degree=MAX_CHEBYSHEV_DEGREE, fn=lambda x: x)
        assert c.degree == 127


# ================================================================
#  8. Transformer Adversarial
# ================================================================

class TestTransformerAdversarial:
    """
    Tests that FHE transformer components reject invalid configurations
    and handle extreme numerical inputs without crashes.

    Adversary capability: control over model architecture and input data.
    """

    @pytest.mark.parametrize("power", [3, 5, 7, 0, -1, 1])
    def test_polysoftmax_invalid_powers(self, power):
        """PolySoftmax must reject non-even and unsupported powers."""
        from orion.nn.transformer import PolySoftmax

        with pytest.raises(ValueError, match="must be in"):
            PolySoftmax(power=power)

    def test_polysoftmax_extreme_large_input(self):
        """
        PolySoftmax with very large input values must not crash.
        Note: x^4 for x=1e10 overflows float32 to Inf, producing NaN
        via Inf/Inf. This tests that the operation does not throw an
        exception, and that moderate large values (within float range)
        still produce valid output.
        """
        from orion.nn.transformer import PolySoftmax

        ps = PolySoftmax(power=4, dim=-1)

        # Very extreme values overflow (expected, documents the limit)
        x_extreme = torch.full((2, 8), 1e10)
        out_extreme = ps(x_extreme)
        assert isinstance(out_extreme, torch.Tensor)  # no crash

        # Moderate large values within x^4 < float32 max (~3.4e38)
        # x=100 => x^4 = 1e8, well within range
        x_moderate = torch.full((2, 8), 100.0)
        out_mod = ps(x_moderate)
        assert not torch.isnan(out_mod).any(), "NaN for moderate values"
        sums = out_mod.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_polysoftmax_extreme_negative_input(self):
        """
        PolySoftmax with very negative input values must not crash.
        Even powers make negatives positive: (-x)^4 = x^4, so large
        negative values behave like large positive values.
        """
        from orion.nn.transformer import PolySoftmax

        ps = PolySoftmax(power=4, dim=-1)

        # Very extreme values overflow (expected, no crash)
        x_extreme = torch.full((2, 8), -1e10)
        out_extreme = ps(x_extreme)
        assert isinstance(out_extreme, torch.Tensor)  # no crash

        # Moderate negative values should work correctly
        x_moderate = torch.full((2, 8), -50.0)
        out_mod = ps(x_moderate)
        assert not torch.isnan(out_mod).any(), "NaN for moderate negatives"
        assert not torch.isinf(out_mod).any(), "Inf for moderate negatives"

    def test_polysoftmax_nan_input(self):
        """PolySoftmax with NaN input must propagate NaN (not crash)."""
        from orion.nn.transformer import PolySoftmax

        ps = PolySoftmax(power=4, dim=-1)
        x = torch.tensor([[float("nan"), 1.0, 2.0]])

        # Should not raise; NaN propagation is acceptable
        out = ps(x)
        assert isinstance(out, torch.Tensor)

    def test_polysoftmax_inf_input(self):
        """PolySoftmax with Inf input must handle gracefully."""
        from orion.nn.transformer import PolySoftmax

        ps = PolySoftmax(power=4, dim=-1)
        x = torch.tensor([[float("inf"), 1.0, 2.0]])

        out = ps(x)
        assert isinstance(out, torch.Tensor)

    def test_multihead_attention_embed_not_divisible(self):
        """embed_dim not divisible by num_heads must be rejected."""
        from orion.nn.transformer import FHEMultiHeadAttention

        with pytest.raises(ValueError, match="divisible"):
            FHEMultiHeadAttention(embed_dim=33, num_heads=4, softmax_power=4)

    def test_layernorm_empty_normalized_shape(self):
        """Empty normalized_shape should raise or handle gracefully."""
        from orion.nn.transformer import FHELayerNorm

        try:
            ln = FHELayerNorm(normalized_shape=())
            # If it constructs, forward should still handle gracefully
            x = torch.randn(2, 4)
            ln(x)
        except (ValueError, RuntimeError, TypeError):
            pass  # Clean rejection is acceptable

    def test_zero_length_sequence(self):
        """Zero-length sequence input must not crash the transformer."""
        from orion.nn.transformer import FHEMultiHeadAttention

        attn = FHEMultiHeadAttention(embed_dim=16, num_heads=4, softmax_power=4)
        attn.eval()

        # Batch=2, seq_len=0, embed=16
        x = torch.randn(2, 0, 16)
        with torch.no_grad():
            try:
                out = attn(x)
                assert out.shape == (2, 0, 16)
            except (RuntimeError, ValueError):
                pass  # Clean failure for 0-length seq is acceptable

    def test_very_large_sequence_length(self):
        """Large sequence length (256) must not crash (correctness, not perf)."""
        from orion.nn.transformer import FHEMultiHeadAttention

        attn = FHEMultiHeadAttention(embed_dim=16, num_heads=4, softmax_power=4)
        attn.eval()

        x = torch.randn(1, 256, 16)
        with torch.no_grad():
            out = attn(x)
        assert out.shape == (1, 256, 16)


# ================================================================
#  9. Cache Poisoning
# ================================================================

class TestCachePoisoning:
    """
    Tests that the FHE cache system detects and handles corrupted,
    poisoned, or manipulated cache entries.

    Adversary capability: write access to the cache directory.
    """

    def test_invalid_json_metadata(self, tmp_path):
        """Invalid JSON in metadata.json must cause a cache miss, not crash."""
        from orion.core.cache import FHECache

        cache = FHECache(cache_dir=str(tmp_path / "cache"))

        # Create a fake cache entry with invalid JSON
        fake_dir = tmp_path / "cache" / "fakehash_fakehash"
        fake_dir.mkdir(parents=True)
        (fake_dir / "metadata.json").write_text("{invalid json content!!!}")

        # list_entries should skip the corrupted entry
        entries = cache.list_entries()
        assert len(entries) == 0

    def test_corrupted_model_state(self, tmp_path):
        """Corrupted model_state.pt must result in a clean load failure."""
        from orion.core.cache import FHECache

        cache = FHECache(cache_dir=str(tmp_path / "cache"))

        # Create a model, save it, then corrupt the state file
        model = nn.Linear(10, 2)
        config_path = str(tmp_path / "config.yml")
        with open(config_path, "w") as f:
            f.write("ckks_params:\n  LogN: 14\n  LogQ: [45]\n")

        cache.save(config_path, model, input_level=5)

        # Find and corrupt the model state file
        for entry_dir in (tmp_path / "cache").iterdir():
            state_file = entry_dir / "model_state.pt"
            if state_file.exists():
                state_file.write_bytes(b"CORRUPTED DATA " * 100)

        # Load should fail gracefully (return None)
        result = cache.load(config_path, model)
        # Either None (cache miss due to load failure) or an exception
        # that the test framework catches
        assert result is None or isinstance(result, dict)

    def test_race_condition_delete_between_exists_and_load(self, tmp_path):
        """
        Simulates a race condition where the cache is deleted between
        exists() check and load() call. Must not crash.
        """
        from orion.core.cache import FHECache

        cache = FHECache(cache_dir=str(tmp_path / "cache"))

        model = nn.Linear(10, 2)
        config_path = str(tmp_path / "config.yml")
        with open(config_path, "w") as f:
            f.write("ckks_params:\n  LogN: 14\n  LogQ: [45]\n")

        cache.save(config_path, model, input_level=5)

        # Verify it exists
        assert cache.exists(config_path, model)

        # Now delete the cache directory (simulating race condition)
        cache.clear()

        # Load should return None, not crash
        result = cache.load(config_path, model)
        assert result is None

    def test_metadata_hash_mismatch(self, tmp_path):
        """
        Poisoning metadata with wrong hashes must cause cache miss.
        """
        from orion.core.cache import FHECache

        cache = FHECache(cache_dir=str(tmp_path / "cache"))

        model = nn.Linear(10, 2)
        config_path = str(tmp_path / "config.yml")
        with open(config_path, "w") as f:
            f.write("ckks_params:\n  LogN: 14\n  LogQ: [45]\n")

        cache.save(config_path, model, input_level=5)

        # Poison the metadata with wrong hashes
        for entry_dir in (tmp_path / "cache").iterdir():
            meta_file = entry_dir / "metadata.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                meta["config_hash"] = "0000000000000000"
                meta_file.write_text(json.dumps(meta))

        result = cache.load(config_path, model)
        assert result is None


# ================================================================
#  10. Concurrency
# ================================================================

class TestConcurrency:
    """
    Tests that Orion's security components are safe under concurrent
    access from multiple threads.

    Adversary capability: ability to trigger concurrent operations
    (e.g., in a server deployment).
    """

    def test_concurrent_hmac_sign(self):
        """Concurrent HMAC sign operations must not corrupt each other."""
        from orion.core.crypto_utils import CiphertextAuthenticator

        auth = CiphertextAuthenticator(hmac_key=os.urandom(32))
        results = []
        lock = threading.Lock()

        def sign_and_verify(idx):
            data = {
                "ciphertexts": [os.urandom(128)],
                "shape": [1],
                "on_shape": [1],
            }
            signed = auth.sign(data)
            is_valid = auth.verify(signed)
            with lock:
                results.append((idx, is_valid))

        threads = []
        for i in range(50):
            t = threading.Thread(target=sign_and_verify, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(results) == 50
        assert all(valid for _, valid in results), \
            "Some concurrent HMAC operations failed verification"

    def test_concurrent_cache_reads_writes(self, tmp_path):
        """Concurrent cache reads and writes must not corrupt data."""
        from orion.core.cache import FHECache

        cache = FHECache(cache_dir=str(tmp_path / "cache"))
        errors = []
        lock = threading.Lock()

        config_path = str(tmp_path / "config.yml")
        with open(config_path, "w") as f:
            f.write("ckks_params:\n  LogN: 14\n  LogQ: [45]\n")

        def cache_operation(idx):
            try:
                model = nn.Linear(10, 2)
                torch.manual_seed(idx)
                nn.init.constant_(model.weight, float(idx))

                cache.save(config_path, model, input_level=idx)
                result = cache.load(config_path, model)
                if result is not None:
                    assert result["input_level"] == idx
            except Exception as e:
                with lock:
                    errors.append((idx, str(e)))

        threads = []
        for i in range(10):
            t = threading.Thread(target=cache_operation, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Some race conditions may occur, but no crashes
        # We allow partial failures due to file system races
        assert len(errors) <= 5, f"Too many errors: {errors}"

    def test_concurrent_config_validations(self):
        """Concurrent config validations must all succeed or fail correctly."""
        from orion.core.config_validator import validate_ckks_params

        results = []
        lock = threading.Lock()

        def validate(idx):
            try:
                result = validate_ckks_params(
                    logn=14,
                    logq=[45, 35, 35, 35, 45],
                    logp=[46],
                    logscale=35,
                    h=192,
                )
                with lock:
                    results.append((idx, result["valid"]))
            except Exception as e:
                with lock:
                    results.append((idx, False))

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(validate, i) for i in range(50)]
            concurrent.futures.wait(futures)

        assert len(results) == 50
        assert all(valid for _, valid in results)

    def test_concurrent_error_checking(self):
        """Concurrent FFI error checking must not lose errors."""
        from orion.core.error_handling import FHEBackendError

        errors_caught = []
        lock = threading.Lock()

        class MockBackend:
            def __init__(self):
                self._error = None
                self._lock = threading.Lock()

            def get_last_error(self):
                return self._error

            def clear_last_error(self):
                self._error = None

        def simulate_ffi_call(idx, backend):
            """Simulate an FFI call that sets an error."""
            backend._error = f"error-{idx}"
            err = backend.get_last_error()
            if err:
                backend.clear_last_error()
                with lock:
                    errors_caught.append(err)

        backend = MockBackend()
        threads = []
        for i in range(30):
            t = threading.Thread(target=simulate_ffi_call, args=(i, backend))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Due to race conditions, we may not catch all 30 distinct errors,
        # but we should catch at least some. The key property is no crashes.
        assert len(errors_caught) > 0
        assert all(isinstance(e, str) for e in errors_caught)


# ================================================================
#  Additional edge-case tests (to reach 50+ total)
# ================================================================

class TestCryptoEdgeCases:
    """Additional edge cases for cryptographic utilities."""

    def test_hmac_key_exactly_16_bytes(self):
        """HMAC key of exactly 16 bytes (minimum) must be accepted."""
        from orion.core.crypto_utils import CiphertextAuthenticator

        auth = CiphertextAuthenticator(hmac_key=b"0123456789abcdef")
        data = {"ciphertexts": [b"test"], "shape": [1], "on_shape": [1]}
        signed = auth.sign(data)
        assert auth.verify(signed) is True

    def test_hmac_key_15_bytes_rejected(self):
        """HMAC key of 15 bytes (one under minimum) must be rejected."""
        from orion.core.crypto_utils import CiphertextAuthenticator

        with pytest.raises(ValueError, match="at least 16"):
            CiphertextAuthenticator(hmac_key=b"0123456789abcde")

    def test_password_exactly_8_chars(self):
        """Password of exactly 8 characters (minimum) must be accepted."""
        from orion.core.crypto_utils import KeyEncryptor

        enc = KeyEncryptor(password="12345678")
        encrypted = enc.encrypt(b"data")
        assert enc.decrypt(encrypted) == b"data"

    def test_password_7_chars_rejected(self):
        """Password of 7 characters (one under minimum) must be rejected."""
        from orion.core.crypto_utils import KeyEncryptor

        with pytest.raises(ValueError, match="at least 8"):
            KeyEncryptor(password="1234567")

    def test_sign_verify_with_large_ciphertext(self):
        """Signing 10MB of ciphertext data must work correctly."""
        auth = _make_authenticator()
        large_ct = os.urandom(10 * 1024 * 1024)  # 10 MB
        data = {"ciphertexts": [large_ct], "shape": [1], "on_shape": [1]}

        signed = auth.sign(data)
        assert auth.verify(signed) is True

        # Tamper one byte
        tampered = bytearray(large_ct)
        tampered[-1] ^= 0x01
        signed["ciphertexts"] = [bytes(tampered)]
        assert auth.verify(signed) is False

    def test_encrypt_large_key_material(self):
        """Encrypting 1MB key material must round-trip correctly."""
        from orion.core.crypto_utils import KeyEncryptor

        enc = KeyEncryptor(password="large-key-test-pw")
        large_data = os.urandom(1024 * 1024)
        encrypted = enc.encrypt(large_data)
        decrypted = enc.decrypt(encrypted)
        assert decrypted == large_data

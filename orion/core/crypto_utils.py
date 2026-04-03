"""
Cryptographic utilities for FHE deployment security.

1. Ciphertext Authentication: HMAC-SHA256 on serialized ciphertexts
   to detect tampering during client-server transport.

2. Key Encryption at Rest: AES-256-GCM encryption of secret key files
   to protect keys stored on disk.

Both use only Python stdlib (hmac, hashlib) and the `cryptography`
library (widely available, FIPS-validated primitives). Falls back to
a simpler scheme if `cryptography` is not installed.
"""

import os
import hmac
import json
import hashlib
import base64
import logging
from typing import Optional

logger = logging.getLogger("orion.crypto")


# ================================================================
#  1. Ciphertext Authentication (HMAC-SHA256)
# ================================================================

class CiphertextAuthenticator:
    """
    Authenticates serialized ciphertexts using HMAC-SHA256.

    Prevents tampering with ciphertexts in transit between client
    and server. Both parties share an HMAC key (derived from the
    FHE secret key or exchanged out-of-band).

    Usage:
        auth = CiphertextAuthenticator(hmac_key=shared_secret)

        # Client side: sign before sending
        signed = auth.sign(serialized_data)

        # Server side: verify before processing
        if auth.verify(signed):
            process(signed["data"])
        else:
            reject("Tampered ciphertext!")
    """
    def __init__(self, hmac_key: bytes):
        if len(hmac_key) < 16:
            raise ValueError("HMAC key must be at least 16 bytes.")
        self.hmac_key = hmac_key

    @classmethod
    def from_secret_key(cls, secret_key_bytes: bytes):
        """
        Derive an HMAC key from the FHE secret key.

        Uses HKDF-like derivation: HMAC-SHA256(sk, "orion-ct-auth").
        """
        derived = hmac.new(
            secret_key_bytes[:32],
            b"orion-ct-auth-v1",
            hashlib.sha256,
        ).digest()
        return cls(hmac_key=derived)

    def _compute_mac(self, data: bytes) -> str:
        """Compute HMAC-SHA256 and return hex digest."""
        return hmac.new(self.hmac_key, data, hashlib.sha256).hexdigest()

    def sign(self, serialized_data: dict) -> dict:
        """
        Sign a serialized ciphertext payload.

        Args:
            serialized_data: Output of CipherTensor.serialize().
                Expected keys: ciphertexts, shape, on_shape.

        Returns:
            dict with original data plus "hmac" field.
        """
        # Compute MAC over concatenated ciphertext bytes
        mac_input = b""
        for ct_bytes in serialized_data["ciphertexts"]:
            if isinstance(ct_bytes, str):
                ct_bytes = base64.b64decode(ct_bytes)
            mac_input += ct_bytes

        mac = self._compute_mac(mac_input)

        return {
            **serialized_data,
            "hmac": mac,
            "hmac_algo": "HMAC-SHA256",
        }

    def verify(self, signed_data: dict) -> bool:
        """
        Verify the HMAC on a signed ciphertext payload.

        Args:
            signed_data: dict with "ciphertexts" and "hmac" fields.

        Returns:
            True if MAC is valid, False if tampered.
        """
        expected_mac = signed_data.get("hmac")
        if not expected_mac:
            logger.warning("No HMAC found in payload — unsigned data.")
            return False

        mac_input = b""
        for ct_bytes in signed_data["ciphertexts"]:
            if isinstance(ct_bytes, str):
                ct_bytes = base64.b64decode(ct_bytes)
            mac_input += ct_bytes

        actual_mac = self._compute_mac(mac_input)
        return hmac.compare_digest(actual_mac, expected_mac)


# ================================================================
#  2. Key Encryption at Rest (AES-256-GCM)
# ================================================================

def _get_aes_backend():
    """Try to import cryptography library for AES-GCM."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        return AESGCM
    except ImportError:
        return None


class KeyEncryptor:
    """
    Encrypts FHE secret keys at rest using AES-256-GCM.

    The encryption key is derived from a user-provided password
    using PBKDF2-HMAC-SHA256 with 600,000 iterations.

    File format:
        {
            "version": 1,
            "salt": <base64>,
            "nonce": <base64>,
            "ciphertext": <base64>,
            "kdf": "PBKDF2-HMAC-SHA256",
            "kdf_iterations": 600000
        }

    Usage:
        encryptor = KeyEncryptor(password="strong-passphrase")

        # Encrypt and save
        encryptor.encrypt_to_file(secret_key_bytes, "secret.key.enc")

        # Load and decrypt
        secret_key_bytes = encryptor.decrypt_from_file("secret.key.enc")
    """
    KDF_ITERATIONS = 600_000
    VERSION = 1

    def __init__(self, password: str):
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters.")
        self.password = password.encode("utf-8")

    def _derive_key(self, salt: bytes) -> bytes:
        """Derive a 256-bit key from password + salt using PBKDF2."""
        return hashlib.pbkdf2_hmac(
            "sha256",
            self.password,
            salt,
            self.KDF_ITERATIONS,
            dklen=32,
        )

    def encrypt(self, plaintext: bytes) -> dict:
        """
        Encrypt data using AES-256-GCM.

        Args:
            plaintext: The raw bytes to encrypt.

        Returns:
            dict with salt, nonce, ciphertext (all base64-encoded).
        """
        AESGCM = _get_aes_backend()
        if AESGCM is None:
            return self._encrypt_fallback(plaintext)

        salt = os.urandom(16)
        nonce = os.urandom(12)
        key = self._derive_key(salt)

        aes = AESGCM(key)
        ciphertext = aes.encrypt(nonce, plaintext, None)

        return {
            "version": self.VERSION,
            "salt": base64.b64encode(salt).decode(),
            "nonce": base64.b64encode(nonce).decode(),
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "kdf": "PBKDF2-HMAC-SHA256",
            "kdf_iterations": self.KDF_ITERATIONS,
            "cipher": "AES-256-GCM",
        }

    def decrypt(self, encrypted: dict) -> bytes:
        """
        Decrypt data encrypted with encrypt().

        Args:
            encrypted: dict from encrypt().

        Returns:
            Decrypted plaintext bytes.

        Raises:
            ValueError: If decryption fails (wrong password or tampered data).
        """
        AESGCM = _get_aes_backend()

        salt = base64.b64decode(encrypted["salt"])
        nonce = base64.b64decode(encrypted["nonce"])
        ciphertext = base64.b64decode(encrypted["ciphertext"])
        key = self._derive_key(salt)

        if AESGCM is None:
            return self._decrypt_fallback(key, nonce, ciphertext)

        aes = AESGCM(key)
        try:
            return aes.decrypt(nonce, ciphertext, None)
        except Exception as e:
            raise ValueError(
                "Decryption failed — wrong password or corrupted data."
            ) from e

    def _encrypt_fallback(self, plaintext: bytes) -> dict:
        """
        Fallback encryption using XOR + HMAC when cryptography is unavailable.
        NOT as secure as AES-GCM but provides basic protection.
        """
        logger.warning(
            "cryptography library not installed. Using HMAC-based fallback. "
            "Install 'cryptography' for AES-256-GCM encryption."
        )
        salt = os.urandom(16)
        nonce = os.urandom(16)
        key = self._derive_key(salt)

        # Generate keystream via HMAC chain
        keystream = b""
        block = nonce
        while len(keystream) < len(plaintext):
            block = hmac.new(key, block, hashlib.sha256).digest()
            keystream += block

        # XOR encrypt
        encrypted = bytes(a ^ b for a, b in zip(plaintext, keystream))

        # Append HMAC for authentication
        mac = hmac.new(key, encrypted, hashlib.sha256).digest()

        return {
            "version": self.VERSION,
            "salt": base64.b64encode(salt).decode(),
            "nonce": base64.b64encode(nonce).decode(),
            "ciphertext": base64.b64encode(encrypted + mac).decode(),
            "kdf": "PBKDF2-HMAC-SHA256",
            "kdf_iterations": self.KDF_ITERATIONS,
            "cipher": "HMAC-XOR-FALLBACK",
        }

    def _decrypt_fallback(self, key: bytes, nonce: bytes, ciphertext_with_mac: bytes) -> bytes:
        """Fallback decryption for HMAC-XOR scheme."""
        mac_size = 32
        if len(ciphertext_with_mac) < mac_size:
            raise ValueError("Corrupted ciphertext.")

        encrypted = ciphertext_with_mac[:-mac_size]
        stored_mac = ciphertext_with_mac[-mac_size:]

        # Verify MAC
        expected_mac = hmac.new(key, encrypted, hashlib.sha256).digest()
        if not hmac.compare_digest(stored_mac, expected_mac):
            raise ValueError(
                "Decryption failed — wrong password or corrupted data.")

        # Generate keystream
        keystream = b""
        block = nonce
        while len(keystream) < len(encrypted):
            block = hmac.new(key, block, hashlib.sha256).digest()
            keystream += block

        return bytes(a ^ b for a, b in zip(encrypted, keystream))

    def encrypt_to_file(self, plaintext: bytes, filepath: str):
        """Encrypt data and save to file."""
        encrypted = self.encrypt(plaintext)
        with open(filepath, "w") as f:
            json.dump(encrypted, f, indent=2)
        logger.info("Encrypted key saved to %s (%s)",
                    filepath, encrypted.get("cipher", "unknown"))

    def decrypt_from_file(self, filepath: str) -> bytes:
        """Load and decrypt data from file."""
        with open(filepath, "r") as f:
            encrypted = json.load(f)
        return self.decrypt(encrypted)

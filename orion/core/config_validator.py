"""
CKKS Parameter Security Validator.

Validates FHE parameters against the Homomorphic Encryption Standard
(https://homomorphicencryption.org/standard/) to prevent users from
accidentally deploying with insecure parameters.

Checks:
  1. Minimum ring dimension (LogN >= 10)
  2. Total LogQP vs security level for given LogN
  3. Hamming weight bounds
  4. LogScale sanity (must be < smallest LogQ prime)
"""

import warnings
from typing import List, Optional

# Security tables for RLWE with ternary secret key distribution.
# Maps LogN -> max total LogQP for each security level.
# Source: Lattigo v6 parameter validation (derived from the LWE
# estimator for RLWE with ternary secret, classical security).
# These match the bounds used by Lattigo, HElib, and SEAL.
HE_STANDARD_128 = {
    10: 41,
    11: 100,
    12: 218,
    13: 438,
    14: 883,
    15: 1770,
    16: 3544,
}

HE_STANDARD_192 = {
    10: 27,
    11: 54,
    12: 109,
    13: 218,
    14: 438,
    15: 881,
    16: 1770,
}

HE_STANDARD_256 = {
    10: 19,
    11: 37,
    12: 75,
    13: 152,
    14: 305,
    15: 611,
    16: 1228,
}

SECURITY_TABLES = {
    128: HE_STANDARD_128,
    192: HE_STANDARD_192,
    256: HE_STANDARD_256,
}

MIN_LOGN = 10
MAX_LOGN = 16


class SecurityValidationError(ValueError):
    """Raised when CKKS parameters fail security validation."""
    pass


class SecurityWarning(UserWarning):
    """Issued for parameters that are technically valid but risky."""
    pass


def compute_total_logqp(logq: List[int], logp: List[int]) -> int:
    """Compute total bit-size of all Q and P primes."""
    return sum(logq) + sum(logp)


def estimate_security_level(logn: int, total_logqp: int) -> Optional[int]:
    """
    Estimate the security level achieved by the given parameters.
    Returns 128, 192, 256, or None if below 128-bit security.
    """
    for level in (256, 192, 128):
        table = SECURITY_TABLES[level]
        if logn in table and total_logqp <= table[logn]:
            return level
    return None


def validate_ckks_params(
    logn: int,
    logq: List[int],
    logp: List[int],
    logscale: int,
    h: int = 192,
    boot_logp: Optional[List[int]] = None,
    min_security: int = 128,
    strict: bool = True,
) -> dict:
    """
    Validate CKKS parameters against the HE Standard.

    Args:
        logn: Log2 of ring dimension N
        logq: List of log2 sizes of Q primes
        logp: List of log2 sizes of P (auxiliary) primes
        logscale: Log2 of default scale
        h: Hamming weight of secret key
        boot_logp: Bootstrapping P primes (included in total if present)
        min_security: Minimum acceptable security level (128, 192, or 256)
        strict: If True, raise errors. If False, return warnings.

    Returns:
        dict with security analysis results.

    Raises:
        SecurityValidationError: If parameters fail validation (strict mode).
    """
    errors = []
    warnings_list = []

    # 1. LogN bounds
    if logn < MIN_LOGN:
        errors.append(
            f"LogN={logn} is below minimum {MIN_LOGN}. "
            f"Ring dimension 2^{logn}={2**logn} provides no meaningful security."
        )
    elif logn > MAX_LOGN:
        errors.append(
            f"LogN={logn} exceeds maximum {MAX_LOGN}. "
            f"Ring dimension 2^{logn} is impractically large."
        )

    # 2. LogQ/LogP sanity
    if not logq:
        errors.append("LogQ is empty — no ciphertext modulus defined.")
    if not logp:
        warnings_list.append("LogP is empty — no auxiliary primes for key switching.")

    for i, q in enumerate(logq):
        if q <= 0:
            errors.append(f"LogQ[{i}]={q} must be positive.")
        if q > 60:
            warnings_list.append(
                f"LogQ[{i}]={q} exceeds 60 bits. Most FHE libraries use "
                f"NTT-friendly primes < 62 bits."
            )

    for i, p in enumerate(logp):
        if p <= 0:
            errors.append(f"LogP[{i}]={p} must be positive.")

    # 3. Total bit-size vs security
    # For bootstrapping, the total Q*P budget includes boot_logp
    total_logqp = compute_total_logqp(logq, logp)

    # If bootstrapping uses additional primes, check that budget too
    boot_total = total_logqp
    if boot_logp and boot_logp != logp:
        boot_total = sum(logq) + sum(boot_logp)
        total_logqp = max(total_logqp, boot_total)

    security = estimate_security_level(logn, total_logqp)

    if logn >= MIN_LOGN and logn <= MAX_LOGN:
        if security is None:
            table_128 = HE_STANDARD_128.get(logn, 0)
            errors.append(
                f"INSECURE PARAMETERS: LogN={logn}, total LogQP={total_logqp} "
                f"exceeds 128-bit security bound of {table_128}. "
                f"These parameters provide less than 128-bit security."
            )
        elif security < min_security:
            errors.append(
                f"Insufficient security: LogN={logn}, total LogQP={total_logqp} "
                f"achieves ~{security}-bit security but {min_security}-bit was requested."
            )

    # 4. LogScale sanity
    if logq:
        min_logq = min(logq[1:-1]) if len(logq) > 2 else min(logq)
        if logscale > min_logq:
            errors.append(
                f"LogScale={logscale} exceeds smallest intermediate LogQ prime "
                f"({min_logq}). Scale will not fit in a single prime, causing "
                f"precision loss or errors."
            )
        if logscale < 10:
            warnings_list.append(
                f"LogScale={logscale} is very small. Precision will be ~{logscale-1} "
                f"bits, which may be insufficient for neural network inference."
            )

    # 5. Hamming weight
    if h < 1:
        errors.append(f"Hamming weight h={h} must be positive.")
    elif h < 64:
        warnings_list.append(
            f"Hamming weight h={h} is low. Standard recommendation is h>=128 "
            f"for secret key distribution."
        )

    # 6. Level budget
    if logq:
        effective_levels = len(logq) - 1
        if effective_levels < 2:
            warnings_list.append(
                f"Only {effective_levels} effective level(s). Most neural networks "
                f"require at least 5-10 levels for meaningful inference."
            )

    result = {
        "valid": len(errors) == 0,
        "security_level": security,
        "total_logqp": total_logqp,
        "effective_levels": len(logq) - 1 if logq else 0,
        "errors": errors,
        "warnings": warnings_list,
    }

    if strict and errors:
        raise SecurityValidationError(
            "CKKS parameter validation failed:\n  - " +
            "\n  - ".join(errors)
        )

    for w in warnings_list:
        warnings.warn(w, SecurityWarning, stacklevel=2)

    return result

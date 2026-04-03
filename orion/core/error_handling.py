"""
Automatic FFI error propagation for the Lattigo backend.

Provides a decorator that wraps Go FFI calls to automatically check
for errors after each call, converting silent Go-side failures into
clean Python exceptions.

Usage:
    # Wrap an entire LattigoLibrary instance:
    wrap_ffi_error_checking(lattigo_lib)

    # Or use the decorator on individual methods:
    @check_ffi_error
    def my_go_call(self, *args):
        return self.backend.SomeGoFunction(*args)
"""

import functools
import logging

logger = logging.getLogger("orion.ffi")

# Functions that return sentinel values on error (-1 for IDs, nil/0 for arrays)
SENTINEL_FUNCTIONS = {
    "NewScheme", "Encode", "Encrypt", "Decrypt",
    "EvaluatePolynomial", "EvaluateLinearTransform",
    "GenerateLinearTransform", "GenerateChebyshev", "GenerateMonomial",
    "NewBootstrapper", "Bootstrap",
    "LoadSecretKey", "LoadCiphertext", "LoadRotationKey",
    "LoadPlaintextDiagonal",
    "SerializeCiphertext",
    "Negate", "Rotate", "RotateNew", "Rescale", "RescaleNew",
    "AddScalar", "AddScalarNew", "SubScalar", "SubScalarNew",
    "MulScalarInt", "MulScalarIntNew", "MulScalarFloat", "MulScalarFloatNew",
    "AddPlaintext", "AddPlaintextNew", "SubPlaintext", "SubPlaintextNew",
    "MulPlaintext", "MulPlaintextNew",
    "AddCiphertext", "AddCiphertextNew", "SubCiphertext", "SubCiphertextNew",
    "MulRelinCiphertext", "MulRelinCiphertextNew",
}


class FHEBackendError(RuntimeError):
    """Raised when the Go FHE backend reports an error."""
    def __init__(self, message, function_name=None):
        self.function_name = function_name
        prefix = f"[{function_name}] " if function_name else ""
        super().__init__(f"{prefix}{message}")


def check_ffi_error(func):
    """
    Decorator that checks for Go-side errors after an FFI call.

    Wraps any method that calls the Lattigo backend. After the call
    completes, checks the Go-side lastError and raises FHEBackendError
    if an error occurred.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Clear any stale error before the call
        if hasattr(self, 'backend'):
            self.backend.clear_last_error()
        elif hasattr(self, 'clear_last_error'):
            self.clear_last_error()

        result = func(self, *args, **kwargs)

        # Check for error after the call
        backend = getattr(self, 'backend', self)
        err = backend.get_last_error()
        if err:
            backend.clear_last_error()
            fname = func.__name__
            logger.error("FFI error in %s: %s", fname, err)
            raise FHEBackendError(err, function_name=fname)

        return result
    return wrapper


def wrap_ffi_error_checking(lattigo_lib):
    """
    Wrap all sentinel-returning functions on a LattigoLibrary instance
    with automatic error checking.

    After calling this, any Go function that sets lastError will
    automatically raise FHEBackendError on the Python side.

    Args:
        lattigo_lib: A LattigoLibrary instance with setup_bindings() called.
    """
    for func_name in SENTINEL_FUNCTIONS:
        original = getattr(lattigo_lib, func_name, None)
        if original is None:
            continue

        def make_wrapper(orig_func, name):
            @functools.wraps(orig_func)
            def wrapped(*args, **kwargs):
                lattigo_lib.clear_last_error()
                result = orig_func(*args, **kwargs)

                # Check for sentinel return values
                if isinstance(result, int) and result == -1:
                    err = lattigo_lib.get_last_error()
                    if err:
                        lattigo_lib.clear_last_error()
                        logger.error("FFI error in %s: %s", name, err)
                        raise FHEBackendError(err, function_name=name)

                # Also check lastError even for non-sentinel returns
                err = lattigo_lib.get_last_error()
                if err:
                    lattigo_lib.clear_last_error()
                    logger.error("FFI error in %s: %s", name, err)
                    raise FHEBackendError(err, function_name=name)

                return result
            return wrapped

        setattr(lattigo_lib, func_name, make_wrapper(original, func_name))

    logger.debug("FFI error checking enabled for %d functions",
                 len(SENTINEL_FUNCTIONS))

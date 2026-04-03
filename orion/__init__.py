from .core import (
    init_scheme,
    delete_scheme,
    encode,
    decode,
    encrypt,
    decrypt,
    fit,
    compile,
)

# Production modules
from .core import (
    validate_ckks_params,
    SecurityValidationError,
    FHEBackendError,
    check_ffi_error,
    managed_cipher,
    managed_plain,
    get_memory_stats,
    cleanup_all,
    MemoryTracker,
    FHECache,
    PipelineExecutor,
    BatchProcessor,
    CiphertextAuthenticator,
    KeyEncryptor,
)

# HuggingFace integration
from .integrations import (
    convert_to_orion,
    check_compatibility,
    HFModelConverter,
    FHECompatibilityReport,
)

__version__ = "1.1.0"
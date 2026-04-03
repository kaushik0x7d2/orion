from .orion import scheme

init_scheme = scheme.init_scheme
delete_scheme = scheme.delete_scheme
encode = scheme.encode
decode = scheme.decode
encrypt = scheme.encrypt
decrypt = scheme.decrypt
fit = scheme.fit
compile = scheme.compile

# Production modules
from .config_validator import validate_ckks_params, SecurityValidationError
from .error_handling import FHEBackendError, check_ffi_error
from .memory import (
    managed_cipher, managed_plain,
    get_memory_stats, cleanup_all, MemoryTracker,
)
from .cache import FHECache
from .parallel import PipelineExecutor, BatchProcessor
from .crypto_utils import CiphertextAuthenticator, KeyEncryptor
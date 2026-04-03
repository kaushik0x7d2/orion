"""
Model and key caching for FHE compiled state.

The fit() + compile() pipeline is expensive (30-60s). This module
caches the compiled state to disk so subsequent runs with the same
model and config skip the expensive steps.

Cache structure:
    .orion_cache/
        <config_hash>/
            metadata.json      # config, model hash, timestamps
            model_state.pt     # compiled model state_dict
            compiled_state.pkl # fit/compile artifacts

Usage:
    cache = FHECache(cache_dir=".orion_cache")

    # Try to load from cache
    state = cache.load(config_path, model)
    if state:
        input_level = state["input_level"]
        # Restore model state
    else:
        # Run fit + compile
        orion.fit(model, data)
        input_level = orion.compile(model)
        cache.save(config_path, model, input_level)
"""

import os
import json
import time
import hashlib
import pickle
import logging
from typing import Optional, Any

import torch

logger = logging.getLogger("orion.cache")


def _hash_config(config_path: str) -> str:
    """Generate a stable hash for a config file."""
    with open(config_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _hash_model(model: torch.nn.Module) -> str:
    """Generate a hash of model architecture and weights."""
    h = hashlib.sha256()
    # Hash architecture (class name + str representation)
    h.update(model.__class__.__name__.encode())
    h.update(str(model).encode())
    # Hash weights
    for name, param in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(param.data.cpu().numpy().tobytes())
    return h.hexdigest()[:16]


class FHECache:
    """
    Persistent cache for FHE compiled model state.

    Caches the result of fit() + compile() keyed by config hash
    and model weight hash. Automatically invalidates when either changes.
    """
    def __init__(self, cache_dir: str = ".orion_cache"):
        self.cache_dir = os.path.abspath(cache_dir)

    def _cache_path(self, config_hash: str, model_hash: str) -> str:
        return os.path.join(self.cache_dir, f"{config_hash}_{model_hash}")

    def _metadata_path(self, cache_path: str) -> str:
        return os.path.join(cache_path, "metadata.json")

    def _model_path(self, cache_path: str) -> str:
        return os.path.join(cache_path, "model_state.pt")

    def _compiled_path(self, cache_path: str) -> str:
        return os.path.join(cache_path, "compiled_state.pkl")

    def exists(self, config_path: str, model: torch.nn.Module) -> bool:
        """Check if a valid cache entry exists."""
        try:
            config_hash = _hash_config(config_path)
            model_hash = _hash_model(model)
            cache_path = self._cache_path(config_hash, model_hash)
            meta_path = self._metadata_path(cache_path)

            if not os.path.exists(meta_path):
                return False

            with open(meta_path, "r") as f:
                meta = json.load(f)

            return (meta.get("config_hash") == config_hash and
                    meta.get("model_hash") == model_hash)
        except Exception:
            return False

    def save(
        self,
        config_path: str,
        model: torch.nn.Module,
        input_level: int,
        extra_data: Optional[dict] = None,
    ):
        """
        Save compiled model state to cache.

        Args:
            config_path: Path to the YAML config file.
            model: The compiled model (after fit + compile).
            input_level: The input encryption level from compile().
            extra_data: Optional additional data to cache.
        """
        config_hash = _hash_config(config_path)
        model_hash = _hash_model(model)
        cache_path = self._cache_path(config_hash, model_hash)

        os.makedirs(cache_path, exist_ok=True)

        # Save metadata
        metadata = {
            "config_hash": config_hash,
            "model_hash": model_hash,
            "config_path": os.path.abspath(config_path),
            "model_class": model.__class__.__name__,
            "input_level": input_level,
            "created_at": time.time(),
            "created_at_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self._metadata_path(cache_path), "w") as f:
            json.dump(metadata, f, indent=2)

        # Save model state
        torch.save(model.state_dict(), self._model_path(cache_path))

        # Save extra compiled state
        if extra_data:
            with open(self._compiled_path(cache_path), "wb") as f:
                pickle.dump(extra_data, f)

        logger.info("Cached compiled state to %s", cache_path)

    def load(
        self,
        config_path: str,
        model: torch.nn.Module,
    ) -> Optional[dict]:
        """
        Load compiled state from cache if available.

        Args:
            config_path: Path to the YAML config file.
            model: The model to restore state into.

        Returns:
            dict with "input_level" and optionally "extra_data",
            or None if cache miss.
        """
        try:
            config_hash = _hash_config(config_path)
            model_hash = _hash_model(model)
            cache_path = self._cache_path(config_hash, model_hash)
            meta_path = self._metadata_path(cache_path)

            if not os.path.exists(meta_path):
                logger.debug("Cache miss: no metadata at %s", meta_path)
                return None

            with open(meta_path, "r") as f:
                meta = json.load(f)

            if meta.get("config_hash") != config_hash:
                logger.debug("Cache miss: config hash mismatch")
                return None
            if meta.get("model_hash") != model_hash:
                logger.debug("Cache miss: model hash mismatch")
                return None

            # Restore model state
            model_path = self._model_path(cache_path)
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, weights_only=True)
                model.load_state_dict(state_dict)

            result = {
                "input_level": meta["input_level"],
                "created_at": meta.get("created_at_str", "unknown"),
            }

            # Load extra data
            compiled_path = self._compiled_path(cache_path)
            if os.path.exists(compiled_path):
                with open(compiled_path, "rb") as f:
                    result["extra_data"] = pickle.load(f)

            logger.info("Cache hit: loaded from %s (created %s)",
                       cache_path, result["created_at"])
            return result

        except Exception as e:
            logger.warning("Cache load failed: %s", e)
            return None

    def clear(self):
        """Remove all cached entries."""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            logger.info("Cleared cache at %s", self.cache_dir)

    def list_entries(self) -> list:
        """List all cache entries with metadata."""
        entries = []
        if not os.path.exists(self.cache_dir):
            return entries

        for dirname in os.listdir(self.cache_dir):
            meta_path = os.path.join(self.cache_dir, dirname, "metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    entries.append(meta)
                except Exception:
                    pass
        return entries

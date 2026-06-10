"""Sentence-Transformers embedder with disk cache."""

from __future__ import annotations

import hashlib
import os
import pickle
import threading
from typing import Iterable

import numpy as np


def _safe_model_name(name: str) -> str:
    return name.replace("/", "_")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Embedder:
    """Cached text embedder."""

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: str = ".cache/embeddings",
        device: str | None = None,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self._cache_path = os.path.join(
            cache_dir, f"{_safe_model_name(model_name)}.pkl"
        )
        self._cache: dict[str, np.ndarray] = {}
        self._dirty = False
        self._model = None
        self._lock = threading.Lock()
        os.makedirs(cache_dir, exist_ok=True)
        self._load_cache()

    def _load_cache(self) -> None:
        if not os.path.exists(self._cache_path):
            return
        try:
            with open(self._cache_path, "rb") as f:
                self._cache = pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"[Embedder] failed to load cache {self._cache_path}: {e}")
            self._cache = {}

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dim(self) -> int:
        # Default MiniLM dimension.
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        return 384

    def embed(self, texts: Iterable[str], normalize: bool = True) -> np.ndarray:
        """Return shape (N, dim) float32. L2-normalized by default."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        keys = [_hash_text(t) for t in texts]
        missing_idx = [i for i, k in enumerate(keys) if k not in self._cache]

        if missing_idx:
            with self._lock:
                # Recheck under lock.
                missing_idx = [i for i, k in enumerate(keys) if k not in self._cache]
                if missing_idx:
                    model = self._ensure_model()
                    new_texts = [texts[i] for i in missing_idx]
                    vecs = model.encode(
                        new_texts,
                        normalize_embeddings=normalize,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    ).astype(np.float32)
                    for i, v in zip(missing_idx, vecs):
                        self._cache[keys[i]] = v
                    self._dirty = True

        out = np.stack([self._cache[k] for k in keys], axis=0).astype(np.float32)
        if normalize:
            # Cache stores normalized vectors.
            pass
        return out

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def flush(self) -> None:
        """Flush cache."""
        if not self._dirty:
            return
        tmp = self._cache_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self._cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self._cache_path)
        self._dirty = False

    def __del__(self):
        try:
            self.flush()
        except Exception:
            pass

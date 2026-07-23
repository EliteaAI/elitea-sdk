import hashlib
import os
from collections import OrderedDict
from typing import Optional


DEFAULT_MAX_SIZE = 500
_ENV_VAR = "ELITEA_IMAGE_CACHE_MAX_SIZE"


def _resolve_default_max_size() -> int:
    raw = os.environ.get(_ENV_VAR)
    if raw is None:
        return DEFAULT_MAX_SIZE
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_SIZE
    return parsed if parsed > 0 else DEFAULT_MAX_SIZE


class ImageDescriptionCache:
    """In-memory LRU cache for LLM-generated image descriptions.

    Runtime-instance scoped: each toolkit wrapper holds its own cache, so cache
    hits are shared across all image-description calls made during a single
    indexing run / tool invocation, but not across processes.

    Key = md5(image bytes) [+ image_name] [+ md5(prompt)] — including the
    prompt guards against returning a description generated for a different
    prompt when the same image is re-encountered with new instructions.
    """

    def __init__(self, max_size: Optional[int] = None):
        self.max_size = max_size if max_size and max_size > 0 else _resolve_default_max_size()
        self.cache: "OrderedDict[str, str]" = OrderedDict()

    def _make_key(self, image_data: bytes, image_name: str = "", prompt: str = "") -> str:
        content_hash = hashlib.md5(image_data).hexdigest()
        parts = [content_hash]
        if image_name:
            parts.append(image_name)
        if prompt:
            parts.append(hashlib.md5(prompt.encode("utf-8")).hexdigest())
        return "_".join(parts)

    def get(self, image_data: bytes, image_name: str = "", prompt: str = "") -> Optional[str]:
        if not image_data:
            return None
        key = self._make_key(image_data, image_name, prompt)
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, image_data: bytes, description: str, image_name: str = "", prompt: str = "") -> None:
        if not image_data or not description:
            return
        key = self._make_key(image_data, image_name, prompt)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = description
            return
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = description

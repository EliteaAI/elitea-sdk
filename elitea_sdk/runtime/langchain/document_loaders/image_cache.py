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

    Key = md5(image bytes) [+ md5(prompt)] — the prompt is part of the key so
    that re-encountering the same image with new instructions does not return a
    description generated for a different prompt. ``image_name`` is accepted
    only for logging context and is intentionally NOT part of the key, so that
    identical bytes referenced from different positions (e.g. a company logo
    repeated on every page/slide, or the same asset reused across documents)
    hit the cache.
    """

    def __init__(self, max_size: Optional[int] = None):
        self.max_size = max_size if max_size and max_size > 0 else _resolve_default_max_size()
        self.cache: "OrderedDict[str, str]" = OrderedDict()

    def _make_key(self, image_data: bytes, prompt: str = "") -> str:
        content_hash = hashlib.md5(image_data).hexdigest()
        if not prompt:
            return content_hash
        return f"{content_hash}_{hashlib.md5(prompt.encode('utf-8')).hexdigest()}"

    def get(self, image_data: bytes, image_name: str = "", prompt: str = "") -> Optional[str]:
        if not image_data:
            return None
        key = self._make_key(image_data, prompt)
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, image_data: bytes, description: str, image_name: str = "", prompt: str = "") -> None:
        if not image_data or not description:
            return
        key = self._make_key(image_data, prompt)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = description
            return
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = description

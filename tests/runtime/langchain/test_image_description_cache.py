"""Unit tests for ImageDescriptionCache and perform_llm_prediction_for_image_bytes.

Covers issue #5844: per-toolkit-instance LRU cache for LLM-generated image
descriptions, keyed on md5(bytes) + md5(prompt). ``image_name`` is accepted
by the API for logging context only and is intentionally NOT part of the
key, so that identical bytes referenced from different positions (repeated
logos, reused assets across documents) hit the cache. Also verifies the
shared helper wired through the loaders honours the cache.
"""

import importlib
from unittest.mock import MagicMock

import pytest


def _reload_cache_module(monkeypatch, env_value=None):
    """Reimport image_cache so DEFAULT_MAX_SIZE picks up a fresh env var."""
    import elitea_sdk.runtime.langchain.document_loaders.image_cache as image_cache_module

    if env_value is None:
        monkeypatch.delenv("ELITEA_IMAGE_CACHE_MAX_SIZE", raising=False)
    else:
        monkeypatch.setenv("ELITEA_IMAGE_CACHE_MAX_SIZE", env_value)
    return importlib.reload(image_cache_module)


class TestImageDescriptionCache:
    def test_default_max_size_is_500(self, monkeypatch):
        module = _reload_cache_module(monkeypatch, env_value=None)
        cache = module.ImageDescriptionCache()
        assert cache.max_size == 500

    def test_env_var_overrides_default(self, monkeypatch):
        module = _reload_cache_module(monkeypatch, env_value="7")
        cache = module.ImageDescriptionCache()
        assert cache.max_size == 7

    def test_invalid_env_var_falls_back_to_default(self, monkeypatch):
        module = _reload_cache_module(monkeypatch, env_value="not-a-number")
        cache = module.ImageDescriptionCache()
        assert cache.max_size == 500

    def test_non_positive_env_var_falls_back_to_default(self, monkeypatch):
        module = _reload_cache_module(monkeypatch, env_value="0")
        cache = module.ImageDescriptionCache()
        assert cache.max_size == 500

    def test_ctor_override_wins_over_env(self, monkeypatch):
        module = _reload_cache_module(monkeypatch, env_value="123")
        cache = module.ImageDescriptionCache(max_size=3)
        assert cache.max_size == 3

    def test_set_and_get_roundtrip(self, monkeypatch):
        module = _reload_cache_module(monkeypatch)
        cache = module.ImageDescriptionCache(max_size=8)
        cache.set(b"bytes", "hello", image_name="a.png", prompt="describe")
        assert cache.get(b"bytes", image_name="a.png", prompt="describe") == "hello"

    def test_empty_bytes_never_cache(self, monkeypatch):
        module = _reload_cache_module(monkeypatch)
        cache = module.ImageDescriptionCache()
        cache.set(b"", "no-op", image_name="a", prompt="p")
        assert cache.get(b"", image_name="a", prompt="p") is None

    def test_empty_description_not_stored(self, monkeypatch):
        module = _reload_cache_module(monkeypatch)
        cache = module.ImageDescriptionCache()
        cache.set(b"data", "", image_name="a", prompt="p")
        assert cache.get(b"data", image_name="a", prompt="p") is None

    def test_prompt_is_part_of_key(self, monkeypatch):
        """Same bytes + name, different prompt => must NOT return the stale value."""
        module = _reload_cache_module(monkeypatch)
        cache = module.ImageDescriptionCache()
        cache.set(b"data", "desc-A", image_name="pic", prompt="promptA")
        assert cache.get(b"data", image_name="pic", prompt="promptB") is None
        assert cache.get(b"data", image_name="pic", prompt="promptA") == "desc-A"

    def test_image_name_is_NOT_part_of_key(self, monkeypatch):
        # Same bytes + prompt should hit regardless of image_name — this is
        # what enables cross-doc / cross-position dedup (e.g. a company logo
        # repeated on every slide or reused across documents).
        module = _reload_cache_module(monkeypatch)
        cache = module.ImageDescriptionCache()
        cache.set(b"data", "for-a", image_name="a", prompt="p")
        assert cache.get(b"data", image_name="b", prompt="p") == "for-a"
        assert cache.get(b"data", image_name="", prompt="p") == "for-a"

    def test_lru_evicts_least_recently_used(self, monkeypatch):
        module = _reload_cache_module(monkeypatch)
        cache = module.ImageDescriptionCache(max_size=2)
        cache.set(b"1", "one", image_name="1", prompt="p")
        cache.set(b"2", "two", image_name="2", prompt="p")
        # Touch entry 1 -> now 2 is LRU.
        assert cache.get(b"1", image_name="1", prompt="p") == "one"
        cache.set(b"3", "three", image_name="3", prompt="p")
        # 2 should have been evicted.
        assert cache.get(b"2", image_name="2", prompt="p") is None
        assert cache.get(b"1", image_name="1", prompt="p") == "one"
        assert cache.get(b"3", image_name="3", prompt="p") == "three"

    def test_setting_existing_key_updates_and_promotes(self, monkeypatch):
        module = _reload_cache_module(monkeypatch)
        cache = module.ImageDescriptionCache(max_size=2)
        cache.set(b"1", "one", image_name="a", prompt="p")
        cache.set(b"2", "two", image_name="b", prompt="p")
        # Re-set b"1" -> becomes MRU; adding b"3" should evict b"2".
        cache.set(b"1", "one-v2", image_name="a", prompt="p")
        cache.set(b"3", "three", image_name="c", prompt="p")
        assert cache.get(b"1", image_name="a", prompt="p") == "one-v2"
        assert cache.get(b"2", image_name="b", prompt="p") is None
        assert cache.get(b"3", image_name="c", prompt="p") == "three"


class TestPerformLlmPredictionCacheIntegration:
    def _fresh_helper(self, monkeypatch):
        """Reload image_cache first, then utils, so the helper binds to the reloaded class."""
        import elitea_sdk.runtime.langchain.document_loaders.image_cache as image_cache_module
        import elitea_sdk.runtime.langchain.document_loaders.utils as utils_module

        monkeypatch.delenv("ELITEA_IMAGE_CACHE_MAX_SIZE", raising=False)
        importlib.reload(image_cache_module)
        importlib.reload(utils_module)
        return utils_module, image_cache_module

    def _mock_llm(self, description="a picture"):
        llm = MagicMock()
        result = MagicMock()
        result.content = description
        llm.invoke.return_value = result
        return llm

    def test_no_cache_invokes_llm_and_returns_description(self, monkeypatch):
        utils_module, _ = self._fresh_helper(monkeypatch)
        llm = self._mock_llm("desc")
        out = utils_module.perform_llm_prediction_for_image_bytes(
            b"img-bytes", llm, prompt="what is this",
        )
        assert out == "desc"
        assert llm.invoke.call_count == 1

    def test_cache_miss_invokes_llm_and_populates_cache(self, monkeypatch):
        utils_module, image_cache_module = self._fresh_helper(monkeypatch)
        cache = image_cache_module.ImageDescriptionCache()
        llm = self._mock_llm("desc")

        out = utils_module.perform_llm_prediction_for_image_bytes(
            b"img-bytes", llm, prompt="what is this",
            cache=cache, image_name="pic.png",
        )
        assert out == "desc"
        assert llm.invoke.call_count == 1
        assert cache.get(b"img-bytes", image_name="pic.png", prompt="what is this") == "desc"

    def test_cache_hit_short_circuits_llm(self, monkeypatch):
        utils_module, image_cache_module = self._fresh_helper(monkeypatch)
        cache = image_cache_module.ImageDescriptionCache()
        cache.set(b"img-bytes", "cached-desc", image_name="pic.png", prompt="what is this")
        llm = self._mock_llm("SHOULD-NOT-BE-CALLED")

        out = utils_module.perform_llm_prediction_for_image_bytes(
            b"img-bytes", llm, prompt="what is this",
            cache=cache, image_name="pic.png",
        )
        assert out == "cached-desc"
        assert llm.invoke.call_count == 0

    def test_different_prompt_forces_llm_call(self, monkeypatch):
        """Regression guard: same image + name, new prompt must not hit stale cache."""
        utils_module, image_cache_module = self._fresh_helper(monkeypatch)
        cache = image_cache_module.ImageDescriptionCache()
        cache.set(b"img-bytes", "stale", image_name="pic.png", prompt="promptA")
        llm = self._mock_llm("fresh")

        out = utils_module.perform_llm_prediction_for_image_bytes(
            b"img-bytes", llm, prompt="promptB",
            cache=cache, image_name="pic.png",
        )
        assert out == "fresh"
        assert llm.invoke.call_count == 1

    def test_empty_description_not_stored(self, monkeypatch):
        utils_module, image_cache_module = self._fresh_helper(monkeypatch)
        cache = image_cache_module.ImageDescriptionCache()
        llm = self._mock_llm("")

        out = utils_module.perform_llm_prediction_for_image_bytes(
            b"img-bytes", llm, prompt="p",
            cache=cache, image_name="pic.png",
        )
        assert out == ""
        assert cache.get(b"img-bytes", image_name="pic.png", prompt="p") is None

"""Payload bounding for images sent to vision models.

Every raster image used to be re-encoded to PNG regardless of its source format, so a
large-dimension photo grew ~8x and was rejected by providers for exceeding their limit.
"""
import base64
import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from elitea_sdk.runtime.langchain.tools.utils import (
    BEDROCK_AND_VERTEX_IMAGE_BASE64_LIMIT,
    _LLM_SUPPORTED_IMAGE_FORMATS,
    MAX_IMAGE_BYTES_FOR_LLM,
    MAX_DECODE_PIXELS,
    MAX_IMAGE_DIMENSION_FOR_LLM,
    encode_image_bytes_for_llm,
    encode_image_for_llm,
)

PARTNER_PLATFORM_BASE64_LIMIT = BEDROCK_AND_VERTEX_IMAGE_BASE64_LIMIT


def photo(width, height, seed=0, grain=6):
    """Photographic content: smooth gradients plus grain. Compresses badly as PNG."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:height, 0:width]
    channels = [
        127 + 120 * np.sin(xs / 210.0) * np.cos(ys / 290.0),
        127 + 120 * np.sin((xs + ys) / 250.0),
        127 + 120 * np.cos(xs / 170.0 + ys / 330.0),
    ]
    stacked = np.stack(channels, axis=-1)
    stacked += rng.integers(-grain, grain + 1, stacked.shape)
    return Image.fromarray(np.clip(stacked, 0, 255).astype(np.uint8), "RGB")


def noise(width, height, seed=0):
    """Incompressible content — the only kind that still overruns the budget once
    the image has been capped to MAX_IMAGE_DIMENSION_FOR_LLM."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(
        rng.integers(0, 256, (height, width, 3), dtype=np.uint8), "RGB"
    )


def screenshot(width, height):
    """Flat-region content of the kind PNG compresses well and JPEG would blur."""
    ys, xs = np.mgrid[0:height, 0:width]
    mask = ((xs // 40 + ys // 40) % 2 == 0)[..., None]
    return Image.fromarray(
        np.where(mask, np.uint8(32), np.uint8(250)).repeat(3, axis=2).astype(np.uint8),
        "RGB",
    )


def base64_length(payload):
    return len(base64.b64encode(payload))


class TestPayloadBudget:
    def test_large_photo_fits_provider_limit(self):
        """A 4000x3000 photo used to produce a ~17 MB payload."""
        payload, image_format = encode_image_for_llm(photo(4000, 3000, seed=1))

        assert base64_length(payload) <= PARTNER_PLATFORM_BASE64_LIMIT
        assert len(payload) <= MAX_IMAGE_BYTES_FOR_LLM
        assert image_format in ("png", "jpeg")

    def test_unbounded_png_encoding_would_have_exceeded_the_limit(self):
        source = photo(4000, 3000, seed=1)

        unbounded = io.BytesIO()
        source.save(unbounded, format="PNG")

        assert base64_length(unbounded.getvalue()) > PARTNER_PLATFORM_BASE64_LIMIT
        assert base64_length(encode_image_for_llm(source)[0]) <= PARTNER_PLATFORM_BASE64_LIMIT

    @pytest.mark.parametrize(
        "width,height",
        [(2602, 2724), (4000, 3000), (5600, 3600), (1568, 1568)],
    )
    def test_budget_holds_across_dimensions(self, width, height):
        payload, _ = encode_image_for_llm(photo(width, height, seed=width))

        assert base64_length(payload) <= PARTNER_PLATFORM_BASE64_LIMIT


class TestDimensionCap:
    def test_longest_edge_is_capped(self):
        payload, _ = encode_image_for_llm(photo(4000, 3000, seed=2))

        with Image.open(io.BytesIO(payload)) as encoded:
            assert max(encoded.size) == MAX_IMAGE_DIMENSION_FOR_LLM

    def test_aspect_ratio_is_preserved(self):
        payload, _ = encode_image_for_llm(photo(4000, 1000, seed=3))

        with Image.open(io.BytesIO(payload)) as encoded:
            assert encoded.size == (MAX_IMAGE_DIMENSION_FOR_LLM, MAX_IMAGE_DIMENSION_FOR_LLM // 4)

    def test_small_images_are_not_upscaled(self):
        payload, _ = encode_image_for_llm(screenshot(640, 480))

        with Image.open(io.BytesIO(payload)) as encoded:
            assert encoded.size == (640, 480)


class TestFormatSelection:
    def test_screenshots_stay_lossless_png(self):
        payload, image_format = encode_image_for_llm(screenshot(1200, 900))

        assert image_format == "png"
        with Image.open(io.BytesIO(payload)) as encoded:
            assert encoded.format == "PNG"

    def test_photos_keep_full_resolution_via_jpeg(self):
        """A photo exceeds the budget as PNG, but JPEG holds it at the full
        high-resolution long edge rather than forcing a downscale."""
        payload, image_format = encode_image_for_llm(photo(4000, 3000, seed=4))

        assert image_format == "jpeg"
        with Image.open(io.BytesIO(payload)) as encoded:
            assert max(encoded.size) == MAX_IMAGE_DIMENSION_FOR_LLM
        assert len(payload) <= MAX_IMAGE_BYTES_FOR_LLM

    def test_incompressible_content_falls_back_to_jpeg(self):
        payload, image_format = encode_image_for_llm(noise(2000, 2000, seed=4))

        assert image_format == "jpeg"
        with Image.open(io.BytesIO(payload)) as encoded:
            assert encoded.format == "JPEG"
        assert base64_length(payload) <= PARTNER_PLATFORM_BASE64_LIMIT

    def test_reported_format_matches_the_bytes(self):
        """Providers reject a data URL whose media type disagrees with the payload."""
        for source in (screenshot(800, 600), photo(4000, 3000, seed=5)):
            payload, image_format = encode_image_for_llm(source)
            with Image.open(io.BytesIO(payload)) as encoded:
                assert encoded.format.lower() == image_format


class TestColourModes:
    def test_transparency_is_preserved_when_png_fits(self):
        source = Image.new("RGBA", (300, 300), (255, 0, 0, 0))

        payload, image_format = encode_image_for_llm(source)

        assert image_format == "png"
        with Image.open(io.BytesIO(payload)) as encoded:
            assert encoded.mode == "RGBA"

    def test_transparency_flattens_onto_white_for_jpeg(self):
        """Discarding alpha instead of compositing would render transparent regions
        black, which reads as a redacted image to the model."""
        source = noise(400, 400, seed=17).convert("RGBA")
        source.paste((0, 0, 0, 0), (200, 0, 400, 400))

        payload, image_format = encode_image_for_llm(source, max_bytes=150_000)

        assert image_format == "jpeg"
        with Image.open(io.BytesIO(payload)) as encoded:
            rgb = encoded.convert("RGB")
            width, height = rgb.size
            red, green, blue = rgb.getpixel((width * 3 // 4, height // 2))
            assert min(red, green, blue) > 230

    @pytest.mark.parametrize("mode", ["L", "P", "CMYK", "LA", "1"])
    def test_exotic_modes_encode_without_error(self, mode):
        source = photo(400, 300, seed=7).convert(mode)

        payload, image_format = encode_image_for_llm(source)

        assert payload
        with Image.open(io.BytesIO(payload)) as encoded:
            assert encoded.format.lower() == image_format

    def test_cmyk_is_not_silently_dropped(self):
        """PNG cannot hold CMYK; the encoder must convert rather than raise."""
        payload, _ = encode_image_for_llm(photo(400, 300, seed=8).convert("CMYK"))

        with Image.open(io.BytesIO(payload)) as encoded:
            assert encoded.size == (400, 300)


class TestOverrides:
    def test_max_bytes_override_is_respected(self):
        payload, image_format = encode_image_for_llm(photo(1200, 900, seed=9), max_bytes=80_000)

        assert len(payload) <= 80_000
        assert image_format == "jpeg"

    def test_max_dimension_override_is_respected(self):
        payload, _ = encode_image_for_llm(photo(2000, 2000, seed=10), max_dimension=512)

        with Image.open(io.BytesIO(payload)) as encoded:
            assert max(encoded.size) == 512

    def test_unreachable_budget_returns_best_effort_rather_than_raising(self):
        payload, image_format = encode_image_for_llm(photo(1200, 900, seed=11), max_bytes=1)

        assert image_format == "jpeg"
        assert payload


class TestOversizeIsShedByDownscaling:
    def test_content_too_large_even_as_jpeg_is_downscaled(self):
        payload, image_format = encode_image_for_llm(noise(3000, 3000, seed=12))

        assert image_format == "jpeg"
        with Image.open(io.BytesIO(payload)) as encoded:
            assert max(encoded.size) < MAX_IMAGE_DIMENSION_FOR_LLM
        assert len(payload) <= MAX_IMAGE_BYTES_FOR_LLM

    def test_downscaling_stops_before_destroying_the_image(self, caplog):
        with caplog.at_level("WARNING"):
            payload, _ = encode_image_for_llm(noise(2000, 2000, seed=13), max_bytes=1)

        with Image.open(io.BytesIO(payload)) as encoded:
            assert max(encoded.size) == 256
        assert "provider may reject" in caplog.text

    def test_unreachable_budget_is_reported_not_silent(self, caplog):
        """The old encoder returned an oversized payload with no signal at all."""
        with caplog.at_level("WARNING"):
            encode_image_for_llm(noise(1200, 900, seed=14), max_bytes=1)

        assert caplog.records
        assert caplog.records[-1].levelname == "WARNING"


class TestRawBytesEntryPoint:
    def test_jpeg_source_is_not_mislabelled_as_png(self):
        """Embedded document images used to be sent as image/png whatever they were."""
        source = io.BytesIO()
        photo(1200, 900, seed=15).save(source, format="JPEG", quality=90)
        original = source.getvalue()

        payload, image_format = encode_image_bytes_for_llm(original)

        with Image.open(io.BytesIO(payload)) as encoded:
            assert encoded.format.lower() == image_format
        assert len(payload) <= len(original)

    def test_oversized_embedded_image_is_bounded(self):
        source = io.BytesIO()
        noise(4000, 4000, seed=16).save(source, format="PNG")

        payload, _ = encode_image_bytes_for_llm(source.getvalue())

        assert base64_length(payload) <= PARTNER_PLATFORM_BASE64_LIMIT


class TestProviderLimits:
    def test_dimension_matches_the_high_resolution_tier(self):
        assert MAX_IMAGE_DIMENSION_FOR_LLM == 2576

    def test_budget_targets_the_partner_platform_floor(self):
        assert PARTNER_PLATFORM_BASE64_LIMIT == 5 * 1024 * 1024
        assert MAX_IMAGE_BYTES_FOR_LLM * 4 // 3 < PARTNER_PLATFORM_BASE64_LIMIT


def encoded_bytes(image, image_format, **params):
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, **params)
    return buffer.getvalue()


class TestConformantSourcesAreNotReencoded:
    """Wrapping an already-lossy image in PNG cannot recover detail, only inflate it."""

    @pytest.mark.parametrize("width,height", [(1200, 900), (2000, 1500), (2576, 1932)])
    def test_jpeg_within_limits_passes_through_byte_identical(self, width, height):
        original = encoded_bytes(photo(width, height, seed=18), "JPEG", quality=85)

        payload, image_format = encode_image_bytes_for_llm(original)

        assert payload == original
        assert image_format == "jpeg"

    def test_png_within_limits_passes_through_byte_identical(self):
        original = encoded_bytes(screenshot(1600, 1200), "PNG")

        payload, image_format = encode_image_bytes_for_llm(original)

        assert payload == original
        assert image_format == "png"

    @pytest.mark.parametrize(
        "maker,image_format,params",
        [
            (lambda: photo(1200, 900, seed=19), "JPEG", {"quality": 85}),
            (lambda: photo(4000, 3000, seed=20), "JPEG", {"quality": 85}),
            (lambda: screenshot(1600, 1200), "PNG", {}),
            (lambda: screenshot(4000, 3000), "PNG", {}),
            (lambda: noise(3000, 3000, seed=21), "PNG", {}),
            (lambda: photo(800, 600, seed=22), "BMP", {}),
        ],
    )
    def test_no_usable_source_is_ever_inflated(self, maker, image_format, params):
        original = encoded_bytes(maker(), image_format, **params)

        payload, _ = encode_image_bytes_for_llm(original)

        assert len(payload) <= len(original)

    def test_oversized_jpeg_is_reencoded_as_jpeg_not_png(self):
        original = encoded_bytes(photo(4000, 3000, seed=23), "JPEG", quality=85)

        payload, image_format = encode_image_bytes_for_llm(original)

        assert image_format == "jpeg"
        assert len(payload) < len(original)

    def test_lossy_hint_skips_the_png_attempt(self):
        source = photo(1200, 900, seed=24)

        as_png, png_format = encode_image_for_llm(source, source_format="png")
        as_jpeg, jpeg_format = encode_image_for_llm(source, source_format="jpeg")

        assert png_format == "png"
        assert jpeg_format == "jpeg"
        assert len(as_jpeg) < len(as_png)


class TestUnsupportedSourceFormats:
    """PyMuPDF hands back jpx/jb2/tiff/bmp for embedded PDF images; providers accept
    only JPEG, PNG, GIF and WebP."""

    @pytest.mark.parametrize("image_format", ["BMP", "TIFF", "PPM"])
    def test_converted_to_a_format_the_provider_accepts(self, image_format):
        original = encoded_bytes(photo(600, 400, seed=25), image_format)

        payload, reported = encode_image_bytes_for_llm(original)

        assert reported in _LLM_SUPPORTED_IMAGE_FORMATS
        with Image.open(io.BytesIO(payload)) as encoded:
            assert encoded.format.lower() == reported


class TestMultiFrameSources:
    """Providers use only the first frame, so shipping the rest is pure waste."""

    def make_animation(self, frames, image_format="GIF"):
        buffer = io.BytesIO()
        head, *tail = [frame.convert("P", palette=Image.ADAPTIVE) for frame in frames]
        head.save(buffer, format=image_format, save_all=True, append_images=tail, duration=100)
        return buffer.getvalue()

    def test_animated_gif_is_reduced_to_the_frame_the_model_reads(self):
        original = self.make_animation(
            [noise(500, 400, seed=n) for n in range(30)]
        )

        payload, image_format = encode_image_bytes_for_llm(original)

        with Image.open(io.BytesIO(payload)) as encoded:
            assert getattr(encoded, "n_frames", 1) == 1
        assert image_format in _LLM_SUPPORTED_IMAGE_FORMATS
        assert len(payload) < len(original)

    def test_still_gif_still_passes_through_byte_identical(self):
        buffer = io.BytesIO()
        screenshot(600, 400).convert("P", palette=Image.ADAPTIVE).save(buffer, format="GIF")
        original = buffer.getvalue()

        payload, image_format = encode_image_bytes_for_llm(original)

        assert payload == original
        assert image_format == "gif"


class TestOnDiskSourcesReachThePassthrough:
    """The file_path route had no original bytes, so every image was re-encoded —
    the one place a lossy WebP could still be inflated into PNG."""

    def encode_from_disk(self, image, image_format, suffix, **params):
        from elitea_sdk.runtime.langchain.document_loaders.EliteAImageLoader import (
            EliteAImageLoader,
        )

        captured = {}

        class CapturingLLM:
            def invoke(self, messages):
                for chunk in messages[0].content:
                    if chunk.get("type") == "image_url":
                        captured["url"] = chunk["image_url"]["url"]

                class Result:
                    content = "described"

                return Result()

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            image.save(handle, format=image_format, **params)
            path = Path(handle.name)
        try:
            original = path.read_bytes()
            EliteAImageLoader(file_path=str(path), llm=CapturingLLM()).get_content()
        finally:
            path.unlink()

        header, encoded = captured["url"].split(",", 1)
        return original, base64.b64decode(encoded), header[len("data:"):-len(";base64")]

    @pytest.mark.parametrize(
        "image_format,suffix,params",
        [("WEBP", ".webp", {"quality": 80}), ("JPEG", ".jpg", {"quality": 85})],
    )
    def test_lossy_on_disk_source_is_not_reencoded(self, image_format, suffix, params):
        original, sent, mime = self.encode_from_disk(
            photo(1200, 900, seed=26), image_format, suffix, **params
        )

        assert sent == original
        assert mime == f"image/{image_format.lower()}"

    def test_png_on_disk_source_is_not_reencoded(self):
        original, sent, mime = self.encode_from_disk(screenshot(1600, 1200), "PNG", ".png")

        assert sent == original
        assert mime == "image/png"

    def test_unreadable_file_degrades_to_reencoding_instead_of_raising(self, monkeypatch, tmp_path):
        from elitea_sdk.runtime.langchain.document_loaders.EliteAImageLoader import (
            EliteAImageLoader,
        )

        target = tmp_path / "shot.png"
        screenshot(800, 600).save(target, format="PNG")
        loader = EliteAImageLoader(file_path=str(target))
        monkeypatch.setattr(
            Path, "read_bytes", lambda self: (_ for _ in ()).throw(OSError("gone"))
        )

        assert loader._raster_source_bytes() is None


class TestDecompressionBombGuard:
    """Loader modules raise Image.MAX_IMAGE_PIXELS process-wide, pushing Pillow's own
    guard out to 600 MP. Everything below that decoded unbounded once the document
    loaders started decoding embedded images."""

    def bomb(self, width, height):
        """Well-formed single-colour PNG: trivial on disk, enormous decoded."""
        import struct
        import zlib

        rows = b"".join(b"\x00" + b"\x00" * (width * 3) for _ in range(height))
        def chunk(tag, data):
            return (struct.pack(">I", len(data)) + tag + data
                    + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))
        return (b"\x89PNG\r\n\x1a\x0a".replace(b"\x0a", b"\n")
                + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(rows, 9))
                + chunk(b"IEND", b""))

    def test_oversized_pixel_count_is_refused(self):
        payload = self.bomb(20000, 15000)

        assert len(payload) < 1_000_000
        with pytest.raises(ValueError, match="decode limit"):
            encode_image_bytes_for_llm(payload)

    def test_refusal_names_the_dimensions(self):
        with pytest.raises(ValueError, match=r"20000x15000"):
            encode_image_bytes_for_llm(self.bomb(20000, 15000))

    def test_guard_sits_below_pillows_own_ceiling(self):
        from PIL import Image as PILImageModule
        import elitea_sdk.runtime.langchain.document_loaders.EliteAImageLoader  # noqa: F401

        assert MAX_DECODE_PIXELS < (PILImageModule.MAX_IMAGE_PIXELS or 0) * 2

    @pytest.mark.parametrize("width,height", [(8000, 4000), (5000, 4000), (4000, 3000)])
    def test_legitimate_large_images_still_process(self, width, height):
        payload, image_format = encode_image_bytes_for_llm(self.bomb(width, height))

        assert payload
        assert image_format in _LLM_SUPPORTED_IMAGE_FORMATS

    def test_limit_is_overridable(self):
        source = encoded_bytes(screenshot(800, 600), "PNG")

        with pytest.raises(ValueError, match="decode limit"):
            encode_image_bytes_for_llm(source, max_pixels=1000)

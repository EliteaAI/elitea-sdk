"""
Regression test for issue #6185: read_file returns content_too_large (and
chunking silently breaks) for large JSON/text files whose first non-ASCII
byte falls after chardet's 10 KB sample window.

Root cause: decode_text() sampled only the first _CHARDET_SAMPLE_SIZE bytes
for chardet.detect(). A file that is pure-ASCII in that sample but has a
non-ASCII character later (e.g. an en-dash in prose past byte 10240) got a
confident-but-wrong 'ascii' detection, decode(encoding) then raised, and the
old code turned that into a ValueError instead of falling back to UTF-8 -
kicking the read onto the loader-fallback path, which returns a dict instead
of a str and disables start_line/end_line chunking entirely.
"""
import pytest

from elitea_sdk.tools.utils.text_operations import decode_text


def test_late_nonascii_byte_past_sample_window_still_decodes():
    # First 10 KB pure ASCII, non-ASCII en-dash appears only after that.
    padding = "a" * 10 * 1024
    text = padding + "price – discount"
    data = text.encode("utf-8")

    result = decode_text(data)

    assert result == text


def test_late_nonascii_byte_reproduces_original_6185_shape():
    # Mirrors the exact failure shape from the reported file: ASCII sample,
    # then a UTF-8 multi-byte character (en-dash) further into the file.
    data = (b"x" * 33380) + "LFA MVP 2025 – Figma".encode("utf-8")

    result = decode_text(data)

    assert result.endswith("LFA MVP 2025 – Figma")
    assert len(result) == len(data) - 2  # one 3-byte UTF-8 char -> 1 Python char


def test_early_nonascii_byte_within_sample_window_still_works():
    # Sanity check: non-ASCII within the sample window was never broken.
    text = "price – discount"
    data = text.encode("utf-8")

    assert decode_text(data) == text


def test_undecodable_bytes_still_raise_value_error():
    # Genuine garbage (no confident chardet encoding, invalid UTF-8) must
    # still raise - this fix only removes the false-positive ASCII case.
    data = b"\x80\x81\x82\x83\x84\x85" * 50
    with pytest.raises(ValueError):
        decode_text(data)

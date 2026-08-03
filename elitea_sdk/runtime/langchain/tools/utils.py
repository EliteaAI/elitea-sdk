import base64
import io
import json
import difflib
import logging
import re
import threading

from PIL import Image as PILImage
from PIL.Image import Image
from openpyxl.cell.text import InlineFont
from openpyxl.cell.rich_text import TextBlock, CellRichText

logger = logging.getLogger(__name__)


def tokenize(s):
    return re.split(r'\s+', s)


def untokenize(ts):
    return ' '.join(ts)


def untokenize_cellrichtext(ts):
    result = CellRichText()
    #
    if not ts:
        return result
    #
    result.append(ts[0])
    #
    for item in ts[1:]:
        result.append(' ')
        result.append(item)
    #
    return result


def equalize(s1, s2):
    l1 = tokenize(s1)
    l2 = tokenize(s2)
    res1 = []
    res2 = []
    prev = difflib.Match(0, 0, 0)
    for match in difflib.SequenceMatcher(a=l1, b=l2).get_matching_blocks():
        if prev.a + prev.size != match.a:
            for i in range(prev.a + prev.size, match.a):
                res2 += ['_' * len(l1[i])]
            res1 += l1[prev.a + prev.size:match.a]
        if prev.b + prev.size != match.b:
            for i in range(prev.b + prev.size, match.b):
                res1 += ['_' * len(l2[i])]
            res2 += l2[prev.b + prev.size:match.b]
        res1 += l1[match.a:match.a + match.size]
        res2 += l2[match.b:match.b + match.size]
        prev = match
    return untokenize(res1), untokenize(res2)


def equalize_markdown(s1, s2):
    l1 = tokenize(s1)
    l2 = tokenize(s2)
    res1 = []
    res2 = []
    prev = difflib.Match(0, 0, 0)
    for match in difflib.SequenceMatcher(a=l1, b=l2).get_matching_blocks():
        # Handle removed text in s1
        if prev.a + prev.size != match.a:
            for i in range(prev.a + prev.size, match.a):
                if len(l1[i]):
                    res1 += ['~~' + l1[i] + '~~']  # Removed text marked with strikethrough

        # Handle added text in s2
        if prev.b + prev.size != match.b:
            for i in range(prev.b + prev.size, match.b):
                if len(l2[i]):
                    res2 += ['**' + l2[i] + '**']  # Added text in bold

        # Common text
        res1 += l1[match.a:match.a + match.size]
        res2 += l2[match.b:match.b + match.size]
        prev = match

    return untokenize(res1), untokenize(res2)


def equalize_openpyxl(s1, s2):
    l1 = tokenize(s1)
    l2 = tokenize(s2)
    #
    res1 = []
    res2 = []
    #
    strikethrough = InlineFont(strike=True)
    bold = InlineFont(b=True)
    #
    prev = difflib.Match(0, 0, 0)
    for match in difflib.SequenceMatcher(a=l1, b=l2).get_matching_blocks():
        # Handle removed text in s1
        if prev.a + prev.size != match.a:
            for i in range(prev.a + prev.size, match.a):
                if len(l1[i]):
                    res1.append(TextBlock(strikethrough, l1[i]))  # Removed text
        # Handle added text in s2
        if prev.b + prev.size != match.b:
            for i in range(prev.b + prev.size, match.b):
                if len(l2[i]):
                    res2.append(TextBlock(bold, l2[i]))  # Added text
        # Common text
        res1 += l1[match.a:match.a + match.size]
        res2 += l2[match.b:match.b + match.size]
        prev = match
    #
    return untokenize_cellrichtext(res1), untokenize_cellrichtext(res2)


def replace_source(document, source_replacers, keys=None):
    """ Replace source start(s) """
    if keys is None:
        keys = ["source"]
    #
    for key in keys:
        if key not in document.metadata:
            continue
        #
        document_source = document.metadata[key]
        #
        fixed_source = document_source
        for replace_from, replace_to in source_replacers.items():
            fixed_source = fixed_source.replace(replace_from, replace_to, 1)
        #
        document.metadata[key] = fixed_source


def unpack_json(json_data):
    if (isinstance(json_data, str)):
        if '```json' in json_data:
            json_data = json_data.replace('```json', '').replace('```', '')
            return json.loads(json_data)
        return json.loads(json_data)
    elif (isinstance(json_data, dict)):
        return json_data
    else:
        raise ValueError("Wrong type of json_data")


REQUIRED_NLTK_PACKAGES = (
    "punkt_tab",
    "averaged_perceptron_tagger_eng",
)


def download_nltk(target, force=False):
    """Download only the NLTK resources required by the runtime."""
    from . import state  # pylint: disable=C0415
    #
    if state.nltk_punkt_downloaded and not force:
        return
    #
    import ssl  # pylint: disable=C0415
    #
    try:
        _create_unverified_https_context = ssl._create_unverified_context  # pylint: disable=W0212
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context  # pylint: disable=W0212
    #
    import os  # pylint: disable=C0415
    import nltk  # pylint: disable=C0415,E0401
    import nltk.downloader  # pylint: disable=C0415,E0401
    #
    os.makedirs(target, exist_ok=True)
    #
    nltk.downloader._downloader._download_dir = target  # pylint: disable=W0212
    nltk.data.path = [target]
    #
    for package in REQUIRED_NLTK_PACKAGES:
        nltk.download(package, download_dir=target)
    #
    state.nltk_punkt_downloaded = True

def bytes_to_base64(bt: bytes) -> str:
    return base64.b64encode(bt).decode('utf-8')

def path_to_base64(path) -> str:
    with open(path, 'rb') as binary_file:
        return base64.b64encode(binary_file.read()).decode('utf-8')

HIGH_RESOLUTION_TIER_LONG_EDGE = 2576
MAX_IMAGE_DIMENSION_FOR_LLM = HIGH_RESOLUTION_TIER_LONG_EDGE
PROVIDER_MAX_IMAGE_EDGE = 8000

BEDROCK_AND_VERTEX_IMAGE_BASE64_LIMIT = 5 * 1024 * 1024
_BASE64_SIZE_RATIO = 4 / 3
_MESSAGE_ENVELOPE_HEADROOM_RATIO = 0.94
MAX_IMAGE_BYTES_FOR_LLM = int(
    BEDROCK_AND_VERTEX_IMAGE_BASE64_LIMIT
    / _BASE64_SIZE_RATIO
    * _MESSAGE_ENVELOPE_HEADROOM_RATIO
)

_JPEG_QUALITY = 85
_DOWNSCALE_STEP_RATIO = 0.8
_MIN_DOWNSCALE_DIMENSION = 256

_PNG_SAFE_MODES = frozenset({'1', 'L', 'LA', 'P', 'RGB', 'RGBA'})

_LLM_SUPPORTED_IMAGE_FORMATS = frozenset({'jpeg', 'png', 'gif', 'webp'})
_LOSSY_IMAGE_FORMATS = frozenset({'jpeg'})

# Loader modules raise Image.MAX_IMAGE_PIXELS process-wide, which pushes Pillow's own
# bomb guard out to 600 MP — far past the point where an RGBA decode exhausts memory.
# 36 MP covers a 600 DPI A4 scan while capping worst-case decode at ~145 MB per image.
MAX_DECODE_PIXELS = 36_000_000


def _downscale_to_fit(image: Image, max_dimension: int) -> Image:
    longest = max(image.size)
    if longest <= max_dimension:
        return image
    scale = max_dimension / longest
    width, height = image.size
    return image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        PILImage.LANCZOS,
    )


def _save_to_bytes(image: Image, image_format: str, **params) -> bytes:
    raw_bytes = io.BytesIO()
    image.save(raw_bytes, format=image_format, **params)
    return raw_bytes.getvalue()


def _flatten_onto_white(image: Image) -> Image:
    if image.mode == 'RGB':
        return image
    rgba = image.convert('RGBA')
    canvas = PILImage.new('RGB', rgba.size, (255, 255, 255))
    canvas.paste(rgba, mask=rgba.getchannel('A'))
    return canvas


def encode_image_for_llm(
    image: Image,
    max_dimension: int = MAX_IMAGE_DIMENSION_FOR_LLM,
    max_bytes: int = MAX_IMAGE_BYTES_FOR_LLM,
    source_format: str = None,
) -> tuple[bytes, str]:
    """Returns the encoded bytes and the format they are in, which the caller must
    declare in the ``data:image/<format>;base64,`` URL.

    A lossy source is re-encoded lossily: wrapping an already-quantised image in PNG
    cannot recover the discarded detail and costs roughly 9x its size. Oversized
    payloads are then shed by downscaling rather than by compressing harder, because
    repeated JPEG passes degrade text legibility far faster than a resize.
    """
    source_format = (source_format or image.format or '').lower()
    image = _downscale_to_fit(image, max_dimension)

    if source_format not in _LOSSY_IMAGE_FORMATS:
        png = _save_to_bytes(
            image if image.mode in _PNG_SAFE_MODES else image.convert('RGBA'), 'PNG'
        )
        if len(png) <= max_bytes:
            return png, 'png'

    flattened = _flatten_onto_white(image)
    while True:
        jpeg = _save_to_bytes(flattened, 'JPEG', quality=_JPEG_QUALITY, optimize=True)
        if len(jpeg) <= max_bytes:
            return jpeg, 'jpeg'
        if max(flattened.size) <= _MIN_DOWNSCALE_DIMENSION:
            logger.warning(
                "Image still %d bytes at %dx%d, below which downscaling would destroy "
                "the content; sending anyway and the provider may reject it",
                len(jpeg), *flattened.size,
            )
            return jpeg, 'jpeg'
        flattened = _downscale_to_fit(
            flattened,
            max(_MIN_DOWNSCALE_DIMENSION, int(max(flattened.size) * _DOWNSCALE_STEP_RATIO)),
        )


def encode_image_bytes_for_llm(
    image_bytes: bytes,
    max_dimension: int = MAX_IMAGE_DIMENSION_FOR_LLM,
    max_bytes: int = MAX_IMAGE_BYTES_FOR_LLM,
    max_pixels: int = MAX_DECODE_PIXELS,
) -> tuple[bytes, str]:
    """Bounds raw image bytes for a vision model, returning them untouched whenever
    they already conform — any re-encode of a conformant image is a pure loss.

    A usable source is never replaced by a larger payload, so shrinking an image to the
    model's resolution tier can never cost more bytes than sending it as it arrived.
    Multi-frame sources are re-encoded down to the only frame the model reads.

    Raises ValueError for content too large to decode safely, judged from the header
    before any pixels are allocated.
    """
    with PILImage.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
        if width * height > max_pixels:
            raise ValueError(
                f"image is {width}x{height} ({width * height} px), above the "
                f"{max_pixels} px decode limit"
            )
        image.load()
        source_format = (image.format or '').lower()
        source_is_usable = (
            source_format in _LLM_SUPPORTED_IMAGE_FORMATS
            and getattr(image, 'n_frames', 1) == 1
            and len(image_bytes) <= max_bytes
            and max(image.size) <= PROVIDER_MAX_IMAGE_EDGE
        )
        if source_is_usable and max(image.size) <= max_dimension:
            return image_bytes, source_format

        encoded, image_format = encode_image_for_llm(
            image, max_dimension, max_bytes, source_format
        )
        if source_is_usable and len(encoded) > len(image_bytes):
            return image_bytes, source_format
        return encoded, image_format


class LockedIterator:
    """ Make iterator thread-safe """

    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.iterator.__next__()

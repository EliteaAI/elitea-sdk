import logging
import re
import string

from gensim.parsing import remove_stopwords
from PIL import Image

from ..tools.utils import bytes_to_base64
from ..utils import extract_text_from_completion
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# Minimum pixel dimension for images sent to LLM processing.
# Images with max(width, height) below this are upscaled to avoid
# poor LLM results on very small images (icons, thumbnails, etc.).
MIN_IMAGE_DIMENSION_FOR_LLM = 256


def cleanse_data(document: str) -> str:
    # remove numbers
    document = re.sub(r"\d+", " ", document)

    # print_log("\n",document)
    # remove single characters
    document = " ".join([w for w in document.split() if len(w) > 1])

    # remove punctuations and convert characters to lower case
    document = "".join([
        char.lower()
        for char in document
        if char not in string.punctuation
    ])

    # Remove remove all non-alphanumeric characaters
    document = re.sub(r"\W+", " ", document)

    # Remove 'out of the box' stopwords
    document = remove_stopwords(document)
    # print_log("--- rem ",document)

    # Remove custom keywords
    # for kw in custom_kw:
    #     document = document.replace(kw, "")

    return document

def perform_llm_prediction_for_image_bytes(image_bytes: bytes, llm, prompt: str) -> str:
    """Performs LLM prediction for image content."""
    base64_string = bytes_to_base64(image_bytes)
    result = llm.invoke([
        HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_string}"},
                },
            ]
        )
    ])
    return extract_text_from_completion(result)


def ensure_min_image_size(image: Image.Image, min_dim: int = MIN_IMAGE_DIMENSION_FOR_LLM):
    """Upscale a PIL image if its largest dimension is below min_dim. Returns (image, was_scaled)."""
    w, h = image.size
    if max(w, h) >= min_dim:
        return image, False
    scale = min_dim / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    logger.debug("Upscaling small image from %dx%d to %dx%d for LLM processing", w, h, *new_size)
    return image.resize(new_size, Image.LANCZOS), True


def scale_svg_drawing(drawing, min_dim: int = MIN_IMAGE_DIMENSION_FOR_LLM):
    """Scale an svglib drawing so its largest dimension meets min_dim. Returns (drawing, was_scaled)."""
    max_side = max(drawing.width, drawing.height)
    if max_side >= min_dim:
        return drawing, False
    scale_factor = min_dim / max_side
    logger.debug("Scaling SVG drawing from %.0fx%.0f by factor %.2f", drawing.width, drawing.height, scale_factor)
    drawing.width *= scale_factor
    drawing.height *= scale_factor
    drawing.scale(scale_factor, scale_factor)
    return drawing, True


def preprocess_svg_for_rendering(svg_content: bytes) -> bytes:
    """Fix known svglib rendering issues in SVG content.

    Addresses:
    - ``fill="currentColor"`` / ``stroke="currentColor"`` – not resolved by
      svglib; replaced with ``black`` (the CSS default inherited color).
    - ``clip-path="url(#…)"`` attributes – svglib often silently clips away
      all content; the attribute and matching ``<clipPath>`` definitions are
      removed so the paths remain visible.
    """
    text = svg_content.decode("utf-8", errors="replace")
    text = text.replace("currentColor", "black")
    text = re.sub(r'\s*clip-path="url\([^)]*\)"', "", text)
    text = re.sub(r"<clipPath[^>]*>.*?</clipPath>", "", text, flags=re.DOTALL)
    return text.encode("utf-8")


def create_temp_file(file_content: bytes):
    import tempfile

    # Automatic cleanup with context manager
    with tempfile.NamedTemporaryFile(mode='w+b', delete=True) as temp_file:
        # Write data to temp file
        temp_file.write(file_content)
        temp_file.flush()  # Ensure data is written

        # Get the file path for operations
        return temp_file.name

def file_to_bytes(filepath):
    """
    Reads a file and returns its content as a bytes object.

    Args:
        filepath (str): The path to the file.

    Returns:
        bytes: The content of the file as a bytes object.
    """
    try:
        with open(filepath, "rb") as f:
            file_content_bytes = f.read()
        return file_content_bytes
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return None
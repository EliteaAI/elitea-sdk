import logging
from io import BytesIO
from pathlib import Path
from typing import List, Optional

from PIL import Image
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg

from elitea_sdk.tools.chunkers import markdown_chunker
from .utils import perform_llm_prediction_for_image_bytes, ensure_min_image_size, scale_svg_drawing, preprocess_svg_for_rendering
from ..constants import DEFAULT_MULTIMODAL_PROMPT
from ..tools.utils import image_to_byte_array, bytes_to_base64

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = 300_000_000

# Images have no chunk unit, so above this ceiling they can't be read in bounded
# pieces — the guard advertises full_read_allowed=False. Aligns with ~5 MB vision limit.
MAX_IMAGE_READ_BYTES = 5 * 1024 * 1024


class EliteAImageLoader(BaseLoader):
    """Loads image files using an LLM vision model for advanced analysis, including SVG support."""

    @classmethod
    def get_file_metadata(cls, *, filename: str,
                          file_content=None,
                          file_size=None) -> dict:
        """Report image dimensions/bytes; images are read whole (multimodal), no chunking (PRE-12 #5443)."""
        width, height = cls._read_dimensions(filename, file_content)

        effective_size = file_size if file_size is not None else (
            len(file_content) if file_content else None
        )
        oversized = effective_size is not None and effective_size > MAX_IMAGE_READ_BYTES

        dims_note = (
            f"Image is {width}x{height} px. " if width and height else ""
        )
        meta = {
            "unit": None,
            "instruction_for_readFile": {
                "first_class_params": {},
                "notes": (
                    f"{dims_note}Images are read whole via the AI vision model "
                    f"(multimodal) and have no chunking — there are no line/page/row "
                    f"range parameters. read_file returns the model's transcription "
                    f"of the image."
                ),
            },
        }
        if width and height:
            meta["image_width"] = width
            meta["image_height"] = height

        if oversized:
            meta["read_limits"] = {"full_read_allowed": False}
            meta["instruction_for_readFile"]["notes"] = (
                f"{dims_note}This image is {effective_size} bytes, above the "
                f"{MAX_IMAGE_READ_BYTES}-byte read limit. Images have no chunking "
                f"(they are read whole, multimodally), so a bounded read is not "
                f"possible — the full read is refused."
            )
        return meta

    @classmethod
    def _read_dimensions(cls, filename: str, file_content) -> tuple:
        """Return (width, height) in px, or (None, None) on failure/corrupt content."""
        if not file_content:
            return None, None
        try:
            if Path(filename).suffix.lower() == ".svg":
                svg_content = preprocess_svg_for_rendering(file_content)
                drawing = svg2rlg(BytesIO(svg_content))
                return int(drawing.width), int(drawing.height)
            with Image.open(BytesIO(file_content)) as image:
                return image.width, image.height
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to read image dimensions for %s: %s", filename, exc)
            return None, None

    def __init__(self, file_path=None, **kwargs):
        # Handle both positional and keyword arguments for file_path
        if file_path is not None:
            self.file_path = file_path
        elif kwargs.get('path'):
            self.file_path = kwargs['path']
        elif kwargs.get('file_content'):
            self.file_content = kwargs['file_content']
            self.file_name = kwargs['file_name']
        else:
            raise ValueError(
                "Path parameter is required (either as 'file_path' positional argument or 'path' keyword argument)")
        self.llm = kwargs.get('llm', None)
        self.prompt = kwargs.get('prompt') if kwargs.get(
            'prompt') is not None else DEFAULT_MULTIMODAL_PROMPT  # Use provided prompt or default
        self.max_tokens = kwargs.get('max_tokens', 512)
        self.token_overlap = kwargs.get('token_overlap', 10)
        self._original_dimensions = None
        self._was_scaled = False

    def get_content(self):
        """
        Retrieves the text content from the file or in-memory content.

        Depending on the file type (SVG or raster image), processes the file
        using the configured LLM vision model.

        Returns:
            str: Extracted text content from the file.
        """
        try:
            if hasattr(self, 'file_path'):
                # If file_path is provided
                file_path = Path(self.file_path)
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {self.file_path}")

                if file_path.suffix.lower() == '.svg':
                    text_content = self._process_svg(self.file_path)
                else:
                    text_content = self._process_raster_image(self.file_path)

            elif hasattr(self, 'file_content') and hasattr(self, 'file_name'):
                # If file_content and file_name are provided
                file_name = Path(self.file_name)
                if file_name.suffix.lower() == '.svg':
                    text_content = self._process_svg(BytesIO(self.file_content))
                else:
                    text_content = self._process_raster_image(BytesIO(self.file_content))
            else:
                raise ValueError("Either 'file_path' or 'file_content' and 'file_name' must be provided.")

        except ImportError as e:
            raise ImportError(
                f"Error: SVG processing dependencies not installed. Please install svglib and reportlab: {e}")
        except Exception as e:
            raise ValueError(f"Error opening image or processing SVG: {e}")

        if not text_content or not text_content.strip():
            text_content = "[No readable text detected by LLM]"

        return text_content

    def _process_svg(self, svg_source):
        """Processes an SVG file or in-memory SVG content with automatic upscaling for small SVGs."""
        if isinstance(svg_source, (str, Path)):
            with open(svg_source, 'rb') as f:
                svg_content = f.read()
        else:
            svg_content = svg_source.read()
        svg_content = preprocess_svg_for_rendering(svg_content)

        if self.llm:
            return self.__process_svg_with_llm(svg_content, self.llm, self.prompt)
        else:
            drawing = svg2rlg(BytesIO(svg_content))
            self._original_dimensions = (int(drawing.width), int(drawing.height))
            drawing, self._was_scaled = scale_svg_drawing(drawing)
            return ""

    def _process_raster_image(self, image_source):
        """Processes a raster image with automatic upscaling for small images."""
        image = Image.open(image_source)
        self._original_dimensions = image.size
        image, self._was_scaled = ensure_min_image_size(image)
        if self.llm:
            return self.__perform_llm_prediction_for_image(image, self.llm, self.prompt)
        else:
            return ""

    def __perform_llm_prediction_for_image(self, image: Image, llm, prompt: str) -> str:
        """Performs LLM prediction for image content."""
        byte_array = image_to_byte_array(image)
        return perform_llm_prediction_for_image_bytes(byte_array, llm, prompt)

    def __process_svg_with_llm(self, svg_content: bytes, llm, prompt: str) -> str:
        """Processes SVG content using LLM with automatic upscaling for small SVGs."""
        drawing = svg2rlg(BytesIO(svg_content))
        self._original_dimensions = (int(drawing.width), int(drawing.height))
        drawing, self._was_scaled = scale_svg_drawing(drawing)
        image = self.__render_svg_drawing(drawing)
        return self.__perform_llm_prediction_for_image(image, llm, prompt)

    @staticmethod
    def __render_svg_drawing(drawing):
        """Render an svglib drawing to a PIL RGBA Image."""
        img_data = BytesIO()
        renderPM.drawToFile(drawing, img_data, fmt="PNG")
        img_data.seek(0)
        return Image.open(img_data).convert("RGBA")

    def load(self) -> List[Document]:
        """Load text from image using an LLM vision model, supports SVG. Applies token-based chunking when max_tokens > 0."""
        text_content = self.get_content()

        metadata = {
            "source": str(self.file_path if hasattr(self, 'file_path') else self.file_name),
            "processing_method": "llm" if self.llm else "none",
        }
        if self._original_dimensions is not None:
            metadata["original_dimensions"] = list(self._original_dimensions)
        if self._was_scaled:
            metadata["scaled"] = True

        base_doc = Document(page_content=text_content, metadata=metadata)
        if not self.max_tokens or self.max_tokens <= 0:
            return [base_doc]
        chunks = list(markdown_chunker(
            iter([base_doc]),
            config={"max_tokens": self.max_tokens, "token_overlap": self.token_overlap},
        ))
        # If chunking didn't actually split, return the original doc to preserve clean metadata.
        if len(chunks) <= 1:
            return [base_doc]
        return chunks

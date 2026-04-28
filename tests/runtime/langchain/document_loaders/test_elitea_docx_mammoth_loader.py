"""
Pytest tests for EliteADocxMammothLoader.

These tests specifically verify:
1. Image handling with LLM (use_llm=True) properly invokes the LLM with image bytes
2. Image handling falls back to Tesseract OCR when LLM is not available
3. Image handling falls back to "Transcript is not available" when both fail
4. The fix for issue #4794 - ensuring image.open().read() is called correctly

Run:
  pytest tests/runtime/langchain/document_loaders/test_elitea_docx_mammoth_loader.py -v
"""

import io
from unittest.mock import MagicMock, patch, Mock
import pytest


class MockImageFile:
    """Mock for the file-like object returned by mammoth image.open()"""
    def __init__(self, data: bytes):
        self._data = data
        self._position = 0

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            result = self._data[self._position:]
            self._position = len(self._data)
        else:
            result = self._data[self._position:self._position + size]
            self._position += size
        return result

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockMammothImage:
    """Mock for mammoth's Image object that has an open() method returning a file-like object"""
    def __init__(self, data: bytes, content_type: str = "image/png"):
        self._data = data
        self.content_type = content_type
        self.alt_text = None

    def open(self):
        return MockImageFile(self._data)


# Minimal valid PNG header for testing
MINIMAL_PNG_BYTES = (
    b'\x89PNG\r\n\x1a\n'  # PNG signature
    b'\x00\x00\x00\rIHDR'  # IHDR chunk
    b'\x00\x00\x00\x01'    # width: 1
    b'\x00\x00\x00\x01'    # height: 1
    b'\x08\x02'            # bit depth: 8, color type: 2 (RGB)
    b'\x00\x00\x00'        # compression, filter, interlace
    b'\x90wS\xde'          # CRC
    b'\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N'  # minimal IDAT
    b'\x00\x00\x00\x00IEND\xaeB`\x82'  # IEND chunk
)


class TestDocxMammothLoaderImageHandling:
    """Test suite for image handling in EliteADocxMammothLoader"""

    def test_handle_image_with_llm_calls_read_on_file_object(self):
        """
        Test that __handle_image properly reads bytes from the file object.

        This is the core fix for issue #4794 - the image.open() returns a file-like
        object, and we must call .read() on it to get actual bytes before passing
        to perform_llm_prediction_for_image_bytes.
        """
        from elitea_sdk.runtime.langchain.document_loaders.EliteADocxMammothLoader import (
            EliteADocxMammothLoader
        )

        # Create loader with mocked LLM
        loader = EliteADocxMammothLoader(file_path='/tmp/test.docx')
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='LLM generated description')
        loader.llm = mock_llm
        loader.prompt = 'Describe this image'

        # Create mock mammoth image
        test_bytes = MINIMAL_PNG_BYTES
        mock_image = MockMammothImage(test_bytes)

        # Patch the perform_llm_prediction_for_image_bytes function to verify bytes are passed
        with patch(
            'elitea_sdk.runtime.langchain.document_loaders.EliteADocxMammothLoader.perform_llm_prediction_for_image_bytes'
        ) as mock_predict:
            mock_predict.return_value = 'Mocked LLM description'

            # Call the handler
            result = loader._EliteADocxMammothLoader__handle_image(mock_image)

            # Verify the function was called with actual bytes (not file object)
            mock_predict.assert_called_once()
            call_args = mock_predict.call_args[0]

            # First argument should be bytes
            assert isinstance(call_args[0], bytes), \
                f"Expected bytes, got {type(call_args[0])}. This would fail before the fix!"
            assert call_args[0] == test_bytes, "Bytes should match the test image data"

            # Second argument should be the LLM
            assert call_args[1] == mock_llm

            # Third argument should be the prompt
            assert call_args[2] == 'Describe this image'

            # Result should contain the LLM description
            assert result == {'src': 'Mocked LLM description'}

    def test_handle_image_with_llm_returns_description(self):
        """Test that LLM-generated descriptions are returned correctly"""
        from elitea_sdk.runtime.langchain.document_loaders.EliteADocxMammothLoader import (
            EliteADocxMammothLoader
        )

        loader = EliteADocxMammothLoader(file_path='/tmp/test.docx')
        mock_llm = MagicMock()
        loader.llm = mock_llm
        loader.prompt = 'Describe this image'

        mock_image = MockMammothImage(MINIMAL_PNG_BYTES)

        with patch(
            'elitea_sdk.runtime.langchain.document_loaders.EliteADocxMammothLoader.perform_llm_prediction_for_image_bytes'
        ) as mock_predict:
            mock_predict.return_value = 'A chart showing quarterly revenue growth'

            result = loader._EliteADocxMammothLoader__handle_image(mock_image)

            assert result['src'] == 'A chart showing quarterly revenue growth'

    def test_handle_image_without_llm_uses_tesseract(self):
        """Test that OCR is used when LLM is not available"""
        from elitea_sdk.runtime.langchain.document_loaders.EliteADocxMammothLoader import (
            EliteADocxMammothLoader
        )

        loader = EliteADocxMammothLoader(file_path='/tmp/test.docx')
        loader.llm = None  # No LLM
        loader.prompt = None

        mock_image = MockMammothImage(MINIMAL_PNG_BYTES)

        with patch('elitea_sdk.runtime.langchain.document_loaders.EliteADocxMammothLoader.pytesseract') as mock_tesseract, \
             patch('elitea_sdk.runtime.langchain.document_loaders.EliteADocxMammothLoader.Image') as mock_pil:
            mock_tesseract.image_to_string.return_value = 'OCR extracted text'
            mock_pil.open.return_value = MagicMock()

            result = loader._EliteADocxMammothLoader__handle_image(mock_image)

            # Verify OCR was called
            mock_tesseract.image_to_string.assert_called_once()
            assert result['src'] == 'OCR extracted text'

    def test_handle_image_fallback_on_exception(self):
        """Test that fallback message is returned when both LLM and OCR fail"""
        from elitea_sdk.runtime.langchain.document_loaders.EliteADocxMammothLoader import (
            EliteADocxMammothLoader
        )

        loader = EliteADocxMammothLoader(file_path='/tmp/test.docx')
        mock_llm = MagicMock()
        loader.llm = mock_llm
        loader.prompt = 'Describe this image'

        mock_image = MockMammothImage(MINIMAL_PNG_BYTES)

        with patch(
            'elitea_sdk.runtime.langchain.document_loaders.EliteADocxMammothLoader.perform_llm_prediction_for_image_bytes'
        ) as mock_predict:
            mock_predict.side_effect = Exception('LLM service unavailable')

            result = loader._EliteADocxMammothLoader__handle_image(mock_image)

            # Should fall back to default handler
            assert result['src'] == 'Transcript is not available'

    def test_handle_image_bug_reproduction_without_read(self):
        """
        Test that demonstrates the bug behavior - passing file object instead of bytes.

        This test verifies that bytes_to_base64 fails when given a file-like object,
        which was the root cause of issue #4794.
        """
        from elitea_sdk.runtime.langchain.tools.utils import bytes_to_base64

        # Create a file-like object (what image.open() returns)
        file_obj = io.BytesIO(b'test image data')

        # This should fail - demonstrating the bug
        with pytest.raises(TypeError, match="bytes-like object"):
            bytes_to_base64(file_obj)

        # The correct approach is to call .read() first
        file_obj.seek(0)
        actual_bytes = file_obj.read()
        result = bytes_to_base64(actual_bytes)
        assert result == 'dGVzdCBpbWFnZSBkYXRh'  # base64 of 'test image data'


class TestDocxMammothLoaderIntegration:
    """Integration tests for EliteADocxMammothLoader"""

    def test_loader_initialization(self):
        """Test that loader initializes correctly with various kwargs"""
        from elitea_sdk.runtime.langchain.document_loaders.EliteADocxMammothLoader import (
            EliteADocxMammothLoader
        )

        # Test with file_path
        loader = EliteADocxMammothLoader(file_path='/tmp/test.docx')
        assert loader.path == '/tmp/test.docx'
        assert loader.llm is None
        assert loader.prompt is None
        assert loader.max_tokens == 512

        # Test with all kwargs
        mock_llm = MagicMock()
        loader = EliteADocxMammothLoader(
            file_path='/tmp/test.docx',
            llm=mock_llm,
            prompt='Custom prompt',
            max_tokens=1024,
            extract_images=True
        )
        assert loader.llm == mock_llm
        assert loader.prompt == 'Custom prompt'
        assert loader.max_tokens == 1024
        assert loader.extract_images is True

    def test_loader_with_file_content(self):
        """Test that loader can be initialized with in-memory content"""
        from elitea_sdk.runtime.langchain.document_loaders.EliteADocxMammothLoader import (
            EliteADocxMammothLoader
        )

        loader = EliteADocxMammothLoader(
            file_content=b'docx content',
            file_name='test.docx'
        )
        assert loader.file_content == b'docx content'
        assert loader.file_name == 'test.docx'


class TestProcessContentByTypeDocxWithLLM:
    """Test process_content_by_type with .docx files and use_llm=True"""

    def test_use_llm_config_is_properly_passed(self):
        """
        Test that use_llm=True in chunking_config properly enables LLM for docx processing.

        This tests the full flow from process_content_by_type through to the loader.
        """
        from elitea_sdk.tools.utils.content_parser import process_content_by_type
        from elitea_sdk.runtime.langchain.document_loaders.constants import LoaderProperties

        # Minimal valid DOCX is complex to create, so we'll mock the loader
        with patch(
            'elitea_sdk.tools.utils.content_parser.loaders_map'
        ) as mock_loaders_map:
            # Setup mock loader config
            mock_loader_cls = MagicMock()
            mock_loader_instance = MagicMock()
            mock_loader_cls.return_value = mock_loader_instance
            mock_loader_instance.load.return_value = iter([])

            mock_loaders_map.get.return_value = {
                'class': mock_loader_cls,
                'kwargs': {'extract_images': True},
                'is_multimodal_processing': True,
                'allowed_to_override': {
                    LoaderProperties.LLM.value: False,
                    LoaderProperties.PROMPT_DEFAULT.value: False,
                    LoaderProperties.PROMPT.value: "",
                    'mode': 'paged',
                    'max_tokens': 512,
                }
            }

            # Create mock LLM
            mock_llm = MagicMock()

            # Chunking config with use_llm=True
            chunking_config = {
                '.docx': {
                    'max_tokens': 512,
                    'mode': 'paged',
                    'prompt': '',
                    'use_default_prompt': False,
                    'use_llm': True
                }
            }

            # Process content
            list(process_content_by_type(
                b'fake docx content',
                'test.docx',
                llm=mock_llm,
                chunking_config=chunking_config
            ))

            # Verify loader was called with llm parameter
            mock_loader_cls.assert_called_once()
            call_kwargs = mock_loader_cls.call_args[1]

            # The llm should be passed to the loader
            assert 'llm' in call_kwargs, "LLM should be passed to loader when use_llm=True"
            assert call_kwargs['llm'] == mock_llm

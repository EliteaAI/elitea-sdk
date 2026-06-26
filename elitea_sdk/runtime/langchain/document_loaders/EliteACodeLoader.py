import logging
import os
from typing import Iterator, Generator

from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.helpers import detect_file_encodings
from langchain_core.tools import ToolException

from elitea_sdk.tools.utils.text_operations import decode_text

logger = logging.getLogger(__name__)


class EliteACodeLoader(BaseLoader):

    @classmethod
    def get_file_metadata(cls, *, filename: str,
                          file_content=None,
                          file_size=None) -> dict:
        """Report total line count and advertise start_line/end_line (PRE-3 #5434)."""
        total_lines = 0
        if file_content:
            try:
                if isinstance(file_content, (bytes, bytearray)):
                    try:
                        text = file_content.decode("utf-8")
                    except UnicodeDecodeError:
                        text = decode_text(file_content)
                else:
                    text = file_content
                total_lines = len(text.splitlines())
            except Exception as e:  # pylint: disable=broad-except
                logger.warning("Failed to count lines for %s: %s", filename, e)

        range_hint = f"Valid range 1..{total_lines}. " if total_lines else ""
        instruction = {
            "first_class_params": {
                "start_line": (
                    f"integer (1-indexed, inclusive) — first line to read. "
                    f"{range_hint}Omit to read from the beginning."
                ),
                "end_line": (
                    f"integer (1-indexed, inclusive) — last line to read. "
                    f"{range_hint}Omit to read to the end."
                ),
            },
            "notes": (
                "Use start_line/end_line together to read a bounded slice "
                "of a large source file and keep tokens bounded."
            ),
        }
        return {
            "unit": "lines",
            "total_lines": total_lines,
            "instruction_for_readFile": instruction,
        }

    def __init__(self, **kwargs):
        if kwargs.get('file_path'):
            self.file_path = kwargs['file_path']
        elif kwargs.get('file_content'):
            self.file_content = kwargs['file_content']
            self.file_name = kwargs['file_name']
        else:
            raise ToolException("'file_path' or 'file_content' parameter should be provided.")
        self.encoding = kwargs.get('encoding', 'utf-8')
        self.autodetect_encoding = kwargs.get('autodetect_encoding', False)

    def get_content(self):
        text = ""
        try:
            if hasattr(self, 'file_path') and self.file_path:
                with open(self.file_path, encoding=self.encoding) as f:
                    text = f.read()
            elif hasattr(self, 'file_content') and self.file_content:
                text = self.file_content.decode(self.encoding)
            else:
                raise ValueError("Neither file_path nor file_content is provided.")
        except UnicodeDecodeError as e:
            if self.autodetect_encoding:
                if hasattr(self, 'file_path') and self.file_path:
                    detected_encodings = detect_file_encodings(self.file_path)
                    for encoding in detected_encodings:
                        try:
                            with open(self.file_path, encoding=encoding.encoding) as f:
                                text = f.read()
                            break
                        except UnicodeDecodeError:
                            continue
                elif hasattr(self, 'file_content') and self.file_content:
                    text = decode_text(self.file_content)
                else:
                    raise ValueError("Neither file_path nor file_content is provided for encoding detection.")
            else:
                raise RuntimeError(f"Error loading content with encoding {self.encoding}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading content: {e}") from e

        return text

    def lazy_load(self) -> Iterator[Document]:
        from elitea_sdk.tools.chunkers.code.codeparser import parse_code_files_for_db

        text = self.get_content()
        file_path = str(self.file_path) if hasattr(self, 'file_path') else self.file_name
        file_name = os.path.basename(file_path)

        def file_content_generator():
            yield {
                'file_name': file_name,
                'file_content': text,
            }

        yield from parse_code_files_for_db(file_content_generator())

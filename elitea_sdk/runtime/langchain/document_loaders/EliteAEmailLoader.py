# Copyright (c) 2026 EPAM Systems
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Loader for email files (.eml and .msg formats).

Uses unstructured library for parsing email content including:
- Email metadata (from, to, cc, bcc, subject, date)
- Email body (plain text and HTML)
- Nested attachments (optional)
"""

import logging
import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class EliteAEmailLoader(BaseLoader):
    """
    Loader for email files (.eml and .msg formats).

    Extracts email content and metadata using the unstructured library.
    Supports processing nested attachments when process_attachments=True.
    """

    def __init__(self, **kwargs):
        """
        Initialize EliteAEmailLoader.

        Args:
            **kwargs: Keyword arguments including:
                file_path (str): Path to the email file (.eml or .msg). Required if file_content not provided.
                file_content (bytes): Raw file content. Required if file_path not provided.
                file_name (str): Original filename (used with file_content for format detection).
                process_attachments (bool): Whether to process nested attachments. Default: True.
                ignore_empty_body (bool): Return empty list if email has no body content. Default: True.
                max_tokens (int): Maximum tokens per chunk. Default: 512.

        Raises:
            ValueError: If neither file_path nor (file_content + file_name) is provided.
        """
        self.file_path = kwargs.get('file_path')
        self.file_content = kwargs.get('file_content')
        self.file_name = kwargs.get('file_name')
        self.process_attachments = kwargs.get('process_attachments', True)
        self.ignore_empty_body = kwargs.get('ignore_empty_body', True)
        self.max_tokens = kwargs.get('max_tokens', 512)

        if not self.file_path and not (self.file_content and self.file_name):
            raise ValueError("Either 'file_path' or ('file_content' and 'file_name') must be provided.")

    def _detect_file_type(self, file_path: Optional[str] = None) -> str:
        """
        Detect email file type from extension.

        Args:
            file_path: Path to check (uses self.file_path or self.file_name if not provided)

        Returns:
            'eml' or 'msg'

        Raises:
            ValueError: If file type cannot be determined or is not supported.
        """
        path = file_path or self.file_path or self.file_name
        if not path:
            raise ValueError("Cannot detect file type: no file path or name provided")

        ext = Path(path).suffix.lower()
        if ext == '.eml':
            return 'eml'
        elif ext == '.msg':
            return 'msg'
        else:
            raise ValueError(f"Unsupported email file type: {ext}. Expected .eml or .msg")

    def _parse_eml(self, file_obj) -> List[Document]:
        """
        Parse .eml file using unstructured.

        Args:
            file_obj: File-like object or path to .eml file

        Returns:
            List of Documents with email content and metadata.
        """
        from unstructured.partition.email import partition_email

        kwargs = {'process_attachments': False}

        if hasattr(file_obj, 'read'):
            elements = partition_email(file=file_obj, **kwargs)
        else:
            elements = partition_email(filename=str(file_obj), **kwargs)

        return self._elements_to_documents(elements)

    def _parse_msg(self, file_obj) -> List[Document]:
        """
        Parse .msg file using unstructured.

        Args:
            file_obj: File-like object or path to .msg file

        Returns:
            List of Documents with email content and metadata.
        """
        from unstructured.partition.msg import partition_msg

        kwargs = {'process_attachments': False}

        if hasattr(file_obj, 'read'):
            elements = partition_msg(file=file_obj, **kwargs)
        else:
            elements = partition_msg(filename=str(file_obj), **kwargs)

        return self._elements_to_documents(elements)

    def _extract_headers_from_eml(self) -> dict:
        """
        Extract email headers directly using Python's email module.

        This is used as a fallback when unstructured returns no elements
        (e.g., email with headers but empty body).

        Returns:
            Dict with 'from', 'to', 'cc', 'subject' keys (if present).
        """
        import email
        from email import policy

        headers = {}
        try:
            if self.file_path:
                with open(self.file_path, 'rb') as f:
                    msg = email.message_from_binary_file(f, policy=policy.default)
            else:
                msg = email.message_from_bytes(self.file_content, policy=policy.default)

            if msg['From']:
                headers['from'] = [str(msg['From'])]
            if msg['To']:
                headers['to'] = [str(msg['To'])]
            if msg['Cc']:
                headers['cc'] = [str(msg['Cc'])]
            if msg['Subject']:
                headers['subject'] = str(msg['Subject'])
            if msg['Date']:
                headers['date'] = str(msg['Date'])
        except Exception as e:
            logger.warning(f"Failed to extract email headers: {e}")

        return headers

    def _extract_attachments_from_eml(self) -> List[str]:
        """
        Extract attachment filenames from email using Python's email module.

        Returns:
            List of attachment filenames.
        """
        attachments_with_content = self._extract_attachments_with_content_from_eml()
        return [filename for filename, _ in attachments_with_content]

    def _extract_attachments_with_content_from_eml(self) -> List[Tuple[str, bytes]]:
        """
        Extract attachment filenames and content from email using Python's email module.

        Returns:
            List of tuples (filename, content_bytes).
        """
        import email
        from email import policy

        attachments = []
        try:
            if self.file_path:
                with open(self.file_path, 'rb') as f:
                    msg = email.message_from_binary_file(f, policy=policy.default)
            else:
                msg = email.message_from_bytes(self.file_content, policy=policy.default)

            for part in msg.walk():
                content_disposition = part.get_content_disposition()
                filename = part.get_filename()
                if filename and (content_disposition == 'attachment' or content_disposition != 'inline'):
                    content = part.get_payload(decode=True)
                    if content:
                        attachments.append((filename, content))
        except Exception as e:
            logger.warning(f"Failed to extract email attachments: {e}")

        return attachments

    def _extract_attachments_with_content_from_msg(self) -> List[Tuple[str, bytes]]:
        """
        Extract attachment filenames and content from MSG file using extract_msg library.

        Returns:
            List of tuples (filename, content_bytes).
        """
        import extract_msg

        attachments = []
        msg = None
        try:
            if self.file_path:
                msg = extract_msg.openMsg(self.file_path)
            else:
                # For in-memory content, write to temp file as extract_msg requires file path
                temp_file = None
                try:
                    temp_dir = tempfile.mkdtemp()
                    temp_file = os.path.join(temp_dir, self.file_name or "temp.msg")
                    with open(temp_file, 'wb') as f:
                        f.write(self.file_content)
                    msg = extract_msg.openMsg(temp_file)
                finally:
                    if temp_file and os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                            os.rmdir(os.path.dirname(temp_file))
                        except OSError:
                            pass

            if msg:
                for attachment in msg.attachments:
                    filename = attachment.longFilename or attachment.shortFilename
                    if filename:
                        content = attachment.data
                        if content:
                            attachments.append((filename, content))

        except Exception as e:
            logger.warning(f"Failed to extract MSG attachments: {e}")
        finally:
            if msg:
                try:
                    msg.close()
                except Exception:
                    pass

        return attachments

    def _extract_attachments_with_content(self) -> List[Tuple[str, bytes]]:
        """
        Extract attachment filenames and content from email file.

        Automatically selects the appropriate extraction method based on file type.

        Returns:
            List of tuples (filename, content_bytes).
        """
        try:
            file_type = self._detect_file_type()
            if file_type == 'msg':
                return self._extract_attachments_with_content_from_msg()
            else:
                return self._extract_attachments_with_content_from_eml()
        except ValueError:
            # Fallback to EML extraction if file type detection fails
            return self._extract_attachments_with_content_from_eml()

    def _parse_attachment_content(self, filename: str, content: bytes) -> str:
        """
        Parse attachment content using appropriate loader based on file extension.

        Args:
            filename: The attachment filename.
            content: The attachment content as bytes.

        Returns:
            Parsed content as string, or empty string if parsing fails.
        """
        from .constants import loaders_map

        _, extension = os.path.splitext(filename)
        extension = extension.lower()

        if extension not in loaders_map:
            logger.debug(f"No loader found for extension {extension}, skipping attachment {filename}")
            return ""

        temp_file = None
        try:
            temp_dir = tempfile.mkdtemp()
            temp_file = os.path.join(temp_dir, filename)

            with open(temp_file, 'wb') as f:
                f.write(content)

            loader_config = loaders_map[extension]
            loader_cls = loader_config['class']
            loader_kwargs = loader_config.get('kwargs', {}).copy()

            loader_kwargs.pop('max_tokens', None)

            loader = loader_cls(file_path=temp_file, **loader_kwargs)
            documents = loader.load()

            page_contents = [doc.page_content for doc in documents if doc.page_content]
            return "\n".join(page_contents)

        except Exception as e:
            logger.warning(f"Failed to parse attachment {filename}: {e}")
            return ""
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    os.rmdir(os.path.dirname(temp_file))
                except OSError:
                    pass

    def _elements_to_documents(self, elements) -> List[Document]:
        """
        Convert unstructured elements to LangChain Documents.

        Combines all elements into a single document with structured content
        and extracts email metadata.

        Args:
            elements: List of unstructured Element objects

        Returns:
            List containing a single Document with combined content and metadata.
        """
        is_empty_body = not elements

        # Return early if ignoring empty body emails
        if is_empty_body and self.ignore_empty_body:
            return []

        # Extract metadata based on elements presence
        if is_empty_body:
            # No body content - extract headers directly using email module
            headers = self._extract_headers_from_eml()
            metadata = {**headers}
        else:
            # Extract metadata from unstructured elements
            metadata = {}
            for element in elements:
                elem_metadata = getattr(element, 'metadata', None)
                if elem_metadata:
                    if hasattr(elem_metadata, 'sent_from') and elem_metadata.sent_from:
                        sent_from = elem_metadata.sent_from
                        metadata['from'] = [str(x) for x in sent_from] if isinstance(sent_from, list) else [str(sent_from)]
                    if hasattr(elem_metadata, 'sent_to') and elem_metadata.sent_to:
                        sent_to = elem_metadata.sent_to
                        metadata['to'] = [str(x) for x in sent_to] if isinstance(sent_to, list) else [str(sent_to)]
                    if hasattr(elem_metadata, 'subject') and elem_metadata.subject:
                        metadata['subject'] = str(elem_metadata.subject)
                    if hasattr(elem_metadata, 'cc_recipient') and elem_metadata.cc_recipient:
                        cc = elem_metadata.cc_recipient
                        metadata['cc'] = [str(x) for x in cc] if isinstance(cc, list) else [str(cc)]
                    if hasattr(elem_metadata, 'last_modified') and elem_metadata.last_modified:
                        metadata['date'] = str(elem_metadata.last_modified)
                    if any(k in metadata for k in ['from', 'to', 'subject']):
                        break

            # If date not found from unstructured, extract from email headers
            if 'date' not in metadata:
                headers = self._extract_headers_from_eml()
                if 'date' in headers:
                    metadata['date'] = headers['date']

        # Extract attachments with content for metadata and parsing
        attachments_with_content = self._extract_attachments_with_content()
        attachment_filenames = [filename for filename, _ in attachments_with_content]

        # Add common metadata fields
        metadata['is_empty_body'] = is_empty_body
        metadata['has_attachment'] = len(attachment_filenames) > 0
        if attachment_filenames:
            metadata['attachment'] = attachment_filenames

        # Build header section for page_content
        content_parts = []
        header_parts = []
        if 'subject' in metadata:
            header_parts.append(f"**Subject:** {metadata['subject']}")
        if 'from' in metadata:
            from_str = ', '.join(metadata['from']) if isinstance(metadata['from'], list) else str(metadata['from'])
            header_parts.append(f"**From:** {from_str}")
        if 'to' in metadata:
            to_str = ', '.join(metadata['to']) if isinstance(metadata['to'], list) else str(metadata['to'])
            header_parts.append(f"**To:** {to_str}")
        if 'cc' in metadata:
            cc_str = ', '.join(metadata['cc']) if isinstance(metadata['cc'], list) else str(metadata['cc'])
            header_parts.append(f"**Cc:** {cc_str}")
        if 'date' in metadata:
            header_parts.append(f"**Date:** {metadata['date']}")

        if header_parts:
            content_parts.append('\n'.join(header_parts))
            content_parts.append('\n---\n')

        # Add body content (only for non-empty emails)
        if not is_empty_body:
            body_parts = []
            for element in elements:
                text = str(element).strip()
                if text:
                    body_parts.append(text)
            if body_parts:
                content_parts.append('\n\n'.join(body_parts))

        # Parse and append attachment content if process_attachments is enabled
        if self.process_attachments and attachments_with_content:
            for filename, content in attachments_with_content:
                parsed_content = self._parse_attachment_content(filename, content)
                if parsed_content:
                    content_parts.append(f"\n---\n**Attachment: {filename}**\n\n{parsed_content}")

        page_content = '\n'.join(content_parts)

        return [Document(page_content=page_content, metadata=metadata)]

    def load(self) -> List[Document]:
        """
        Load and parse the email file.

        Returns:
            List of Documents with email content chunked by headers.
        """
        docs = self._load_raw()

        # Apply chunking if max_tokens is set (lazy import to avoid circular dependency)
        if self.max_tokens and self.max_tokens > 0:
            from elitea_sdk.tools.chunkers.sematic.markdown_chunker import markdown_by_headers_chunker
            return list(markdown_by_headers_chunker(
                iter(docs),
                config={'max_tokens': self.max_tokens}
            ))

        return docs

    def _load_raw(self) -> List[Document]:
        """
        Load email content without chunking.

        Returns:
            List of Documents with raw email content.
        """
        file_type = self._detect_file_type()

        if self.file_path:
            # Load from file path
            if file_type == 'eml':
                return self._parse_eml(self.file_path)
            else:
                return self._parse_msg(self.file_path)
        else:
            # Load from memory
            file_obj = BytesIO(self.file_content)
            if file_type == 'eml':
                return self._parse_eml(file_obj)
            else:
                return self._parse_msg(file_obj)

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazily load documents.

        Yields:
            Document objects one at a time.
        """
        for doc in self.load():
            yield doc

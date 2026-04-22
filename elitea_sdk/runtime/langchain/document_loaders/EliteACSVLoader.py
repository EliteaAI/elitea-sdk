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

from io import StringIO
from typing import List, Optional, Iterator, Any
from charset_normalizer import from_path, from_bytes
from csv import DictReader
from langchain_core.documents import Document
from .EliteATableLoader import EliteATableLoader

class EliteACSVLoader(EliteATableLoader):
    def __init__(self,
                 file_path: str = None,
                 file_content: bytes = None,
                 encoding: Optional[str] = 'utf-8',
                 autodetect_encoding: bool = True,
                 json_documents: bool = True,
                 raw_content: bool = False,
                 columns: Optional[List[str]] = None,
                 cleanse: bool = True,
                 **kwargs):
        super().__init__(file_path=file_path, file_content=file_content, json_documents=json_documents, columns=columns, raw_content=raw_content, cleanse=cleanse)
        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding
        self.max_tokens = int(kwargs.get('max_tokens', -1))
        self.add_header_to_chunks = bool(kwargs.get('add_header_to_chunks', False))
        self.header_row_number = max(1, int(kwargs.get('header_row_number', 1))) if kwargs.get('header_row_number') else 1
        if self.file_path:
            if autodetect_encoding:
                self.encoding = from_path(self.file_path).best().encoding
        else:
            self.encoding = from_bytes(self.file_content).best().encoding

    def _chunk_rows(self, rows: List[str]) -> List[str]:
        """Split rows into token-limited chunks using the same algorithm as EliteAExcelLoader._format_sheet_content()."""
        if self.max_tokens < 1:
            return ['\n'.join(rows)]

        import tiktoken
        encoding = tiktoken.get_encoding('cl100k_base')

        def count_tokens(text):
            return len(encoding.encode(text))

        header = None
        if self.add_header_to_chunks and rows:
            header_idx = min(self.header_row_number - 1, len(rows) - 1)
            header = rows.pop(header_idx)

        def finalize_chunk(chunk_rows):
            if self.add_header_to_chunks and header:
                return '\n'.join([header] + chunk_rows)
            return '\n'.join(chunk_rows)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for row in rows:
            row_tokens = count_tokens(row)
            if row_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append(finalize_chunk(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                chunks.append(finalize_chunk([row]) if self.add_header_to_chunks and header else row)
                continue
            if current_tokens + row_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append(finalize_chunk(current_chunk))
                current_chunk = [row]
                current_tokens = row_tokens
            else:
                current_chunk.append(row)
                current_tokens += row_tokens

        if current_chunk:
            chunks.append(finalize_chunk(current_chunk))
        return chunks

    def load(self) -> List[Document]:
        if not self.raw_content:
            return super().load()

        content_list = self.read()
        if not content_list:
            return []

        content = content_list[0]
        if not content:
            return []

        rows = content.splitlines()
        while rows and not rows[-1]:
            rows.pop()

        if not rows:
            return []

        chunks = self._chunk_rows(rows)
        docs = []
        for idx, chunk in enumerate(chunks, start=1):
            metadata = {
                "source": f"{self.file_path}:{idx}",
                "table_source": self.file_path,
                "chunk_id": idx,
            }
            docs.append(Document(page_content=chunk, metadata=metadata))
        return docs

    def read_lazy(self) -> Iterator[dict]:
        with open(self.file_path, 'r', encoding=self.encoding) as fd:
            if self.raw_content:
                content = fd.read()
                if content:
                    yield content
                return
            for row in DictReader(fd):
                yield row

    def read(self) -> Any:
        if self.file_path:
            with open(self.file_path, 'r', encoding=self.encoding) as fd:
                if self.raw_content:
                    content = fd.read()
                    return [content] if content else []
                return list(DictReader(fd))
        else:
            decoded = self.file_content.decode(self.encoding)
            if self.raw_content:
                return [decoded] if decoded else []
            return list(DictReader(StringIO(decoded)))

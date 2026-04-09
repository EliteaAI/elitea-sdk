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

from langchain_community.document_loaders import (
    AirbyteJSONLoader, UnstructuredHTMLLoader,
    UnstructuredXMLLoader, UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader)

from .EliteACSVLoader import EliteACSVLoader
from .EliteADocxMammothLoader import EliteADocxMammothLoader
from .EliteAExcelLoader import EliteAExcelLoader
from .EliteAImageLoader import EliteAImageLoader
from .EliteAJSONLoader import EliteAJSONLoader
from .EliteAJSONLinesLoader import EliteAJSONLinesLoader
from .EliteAPDFLoader import EliteAPDFLoader
from .EliteAPowerPointLoader import EliteAPowerPointLoader
from .EliteATextLoader import EliteATextLoader
from .EliteAMarkdownLoader import EliteAMarkdownLoader
from .EliteAPythonLoader import EliteAPythonLoader
from enum import Enum
from elitea_sdk.runtime.langchain.constants import LOADER_MAX_TOKENS_DEFAULT


class LoaderProperties(Enum):
    LLM = 'use_llm'
    PROMPT_DEFAULT = 'use_default_prompt'
    PROMPT = 'prompt'

DEFAULT_ALLOWED_BASE = {'max_tokens': LOADER_MAX_TOKENS_DEFAULT}

DEFAULT_ALLOWED_WITH_LLM = {
    **DEFAULT_ALLOWED_BASE,
    LoaderProperties.LLM.value: False,
    LoaderProperties.PROMPT_DEFAULT.value: False,
    LoaderProperties.PROMPT.value: "",
}

DEFAULT_ALLOWED_EXCEL = {**DEFAULT_ALLOWED_WITH_LLM, 'add_header_to_chunks': False, 'header_row_number': 1, 'max_tokens': -1, 'sheet_name': ''}

# Image file loaders mapping - directly supported by LLM with image_url
image_loaders_map = {
    '.png': {
        'class': EliteAImageLoader,
        'mime_type': 'image/png',
        'is_multimodal_processing': True,
        'kwargs': {},
        'allowed_to_override': DEFAULT_ALLOWED_WITH_LLM,
    },
    '.jpg': {
        'class': EliteAImageLoader,
        'mime_type': 'image/jpeg',
        'is_multimodal_processing': True,
        'kwargs': {},
        'allowed_to_override': DEFAULT_ALLOWED_WITH_LLM
    },
    '.jpeg': {
        'class': EliteAImageLoader,
        'mime_type': 'image/jpeg',
        'is_multimodal_processing': True,
        'kwargs': {},
        'allowed_to_override': DEFAULT_ALLOWED_WITH_LLM
    },
    '.gif': {
        'class': EliteAImageLoader,
        'mime_type': 'image/gif',
        'is_multimodal_processing': True,
        'kwargs': {},
        'allowed_to_override': DEFAULT_ALLOWED_WITH_LLM
    },
    '.webp': {
        'class': EliteAImageLoader,
        'mime_type': 'image/webp',
        'is_multimodal_processing': True,
        'kwargs': {},
        'allowed_to_override': DEFAULT_ALLOWED_WITH_LLM
    }
}

# Image file loaders mapping - require conversion before sending to LLM
image_loaders_map_converted = {
    '.bmp': {
        'class': EliteAImageLoader,
        'mime_type': 'image/bmp',
        'is_multimodal_processing': True,
        'kwargs': {},
        'allowed_to_override': DEFAULT_ALLOWED_WITH_LLM
    },
    '.svg': {
        'class': EliteAImageLoader,
        'mime_type': 'image/svg+xml',
        'is_multimodal_processing': True,
        'kwargs': {},
        'allowed_to_override': DEFAULT_ALLOWED_WITH_LLM
    }
}

# Document file loaders mapping
document_loaders_map = {
    '.txt': {
        'class': EliteATextLoader,
        'mime_type': 'text/plain',
        'is_multimodal_processing': False,
        'kwargs': {
            'autodetect_encoding': True
        },
        'allowed_to_override': DEFAULT_ALLOWED_BASE
    },
    '.yml': {
        'class': EliteATextLoader,
        'mime_type': 'application/yaml',
        'is_multimodal_processing': False,
        'kwargs': {
            'autodetect_encoding': True
        },
        'allowed_to_override': DEFAULT_ALLOWED_BASE
    },
    '.yaml': {
        'class': EliteATextLoader,
        'mime_type': 'application/yaml',
        'is_multimodal_processing': False,
        'kwargs': {
            'autodetect_encoding': True
        },
        'allowed_to_override': DEFAULT_ALLOWED_BASE
    },
    '.groovy': {
        'class': EliteATextLoader,
        'mime_type': 'text/x-groovy',
        'is_multimodal_processing': False,
        'kwargs': {
            'autodetect_encoding': True
        },
        'allowed_to_override': DEFAULT_ALLOWED_BASE
    },
    '.md': {
        'class': EliteAMarkdownLoader,
        'mime_type': 'text/markdown',
        'is_multimodal_processing': False,
        'kwargs': {},
        'allowed_to_override': DEFAULT_ALLOWED_BASE
    },
    '.csv': {
        'class': EliteACSVLoader,
        'mime_type': 'text/csv',
        'is_multimodal_processing': False,
        'kwargs': {
            'encoding': 'utf-8',
            'raw_content': True,
            'cleanse': False
        },
        'allowed_to_override': DEFAULT_ALLOWED_BASE
    },
    '.xlsx': {
        'class': EliteAExcelLoader,
        'mime_type': ('application/vnd.openxmlformats-officedocument.'
                      'spreadsheetml.sheet'),
        'is_multimodal_processing': False,
        'kwargs': {
            'add_header_to_chunks': False,
            'header_row_number': 1,
            'max_tokens': -1,
            'sheet_name': ''
        },
        'allowed_to_override': DEFAULT_ALLOWED_EXCEL
    },
    '.xls': {
        'class': EliteAExcelLoader,
        'mime_type': 'application/vnd.ms-excel',
        'is_multimodal_processing': False,
        'kwargs': {
            'excel_by_sheets': True,
            'raw_content': True,
            'cleanse': False
        },
        'allowed_to_override': DEFAULT_ALLOWED_EXCEL
    },
    '.pdf': {
        'class': EliteAPDFLoader,
        'mime_type': 'application/pdf',
        'is_multimodal_processing': False,
        'kwargs': {},
        'allowed_to_override': DEFAULT_ALLOWED_WITH_LLM
    },
    '.docx': {
        'class': EliteADocxMammothLoader,
        'mime_type': ('application/vnd.openxmlformats-officedocument.'
                      'wordprocessingml.document'),
        'is_multimodal_processing': True,
        'kwargs': {
            'extract_images': True
        },
        'allowed_to_override': {**DEFAULT_ALLOWED_WITH_LLM, 'mode': 'paged'}
    },
    # Legacy binary Word format — uses unstructured (libreoffice/antiword under the hood)
    '.doc': {
        'class': UnstructuredWordDocumentLoader,
        'mime_type': 'application/msword',
        'is_multimodal_processing': False,
        'kwargs': {},
        'allowed_to_override': DEFAULT_ALLOWED_BASE
    },
    '.json': {
        'class': EliteAJSONLoader,
        'mime_type': 'application/json',
        'is_multimodal_processing': False,
        'kwargs': {},
        'allowed_to_override': DEFAULT_ALLOWED_BASE
    },
    '.jsonl': {
        'class': EliteAJSONLinesLoader,
        'mime_type': 'application/jsonl',
        'is_multimodal_processing': False,
        'kwargs': {},
        'allowed_to_override': DEFAULT_ALLOWED_BASE
    },
    '.htm': {
        'class': UnstructuredHTMLLoader,
        'mime_type': 'text/html',
        'is_multimodal_processing': False,
        'kwargs': {},
        'allowed_to_override': DEFAULT_ALLOWED_WITH_LLM
    },
    '.html': {
        'class': UnstructuredHTMLLoader,
        'mime_type': 'text/html',
        'is_multimodal_processing': False,
        'kwargs': {},
        'allowed_to_override': DEFAULT_ALLOWED_WITH_LLM
    },
    '.xml': {
        'class': UnstructuredXMLLoader,
        'mime_type': 'text/xml',
        'is_multimodal_processing': False,
        'kwargs': {},
        'allowed_to_override': DEFAULT_ALLOWED_WITH_LLM
    },
    # Legacy binary PowerPoint format — uses unstructured (libreoffice under the hood)
    '.ppt': {
        'class': UnstructuredPowerPointLoader,
        'mime_type': 'application/vnd.ms-powerpoint',
        'is_multimodal_processing': False,
        'kwargs': {},
        'allowed_to_override': DEFAULT_ALLOWED_BASE
    },
    '.pptx': {
        'class': EliteAPowerPointLoader,
        'mime_type': ('application/vnd.openxmlformats-officedocument.'
                      'presentationml.presentation'),
        'is_multimodal_processing': False,
        'kwargs': {
            'mode': 'paged'
        },
        'allowed_to_override': {
            **DEFAULT_ALLOWED_WITH_LLM,
            'mode': 'paged',
            'pages_per_chunk': 5,
            'extract_images': False,
        }
    },
    # '.py': {
    #     'class': EliteAPythonLoader,
    #     'mime_type': 'text/x-python',
    #     'is_multimodal_processing': False,
    #     'kwargs': {},
    #     'allowed_to_override': DEFAULT_ALLOWED_BASE
    # }
}

code_extensions = [
    '.py',  # Python
    '.js',  # JavaScript
    '.ts',  # TypeScript
    '.java',  # Java
    '.cpp',  # C++
    '.c',  # C
    '.h',  # C/C++ header
    '.hpp',  # C++ header
    '.cs',  # C#
    '.rb',  # Ruby
    '.go',  # Go
    '.php',  # PHP
    '.swift',  # Swift
    '.kt',  # Kotlin
    '.rs',  # Rust
    '.m',  # Objective-C
    '.scala',  # Scala
    '.pl',  # Perl
    '.sh',  # Shell
    '.bat',  # Batch
    '.lua',  # Lua
    '.r',  # R
    '.pas',  # Pascal
    '.asm',  # Assembly
    '.dart',  # Dart
    '.groovy',  # Groovy
    '.sql',  # SQL
]

default_loader_config = {
    'class': EliteATextLoader,
    'mime_type': 'text/plain',
    'is_multimodal_processing': False,
    'kwargs': {},
    'allowed_to_override': DEFAULT_ALLOWED_BASE
}

code_loaders_map = {ext: default_loader_config for ext in code_extensions}

# Combined mapping for backward compatibility
loaders_map = {
    **image_loaders_map,
    **image_loaders_map_converted,
    **document_loaders_map,
    **code_loaders_map
}

loaders_allowed_to_override = {
    extension: config.get('allowed_to_override')
    for extension, config in loaders_map.items()
    if 'allowed_to_override' in config
}

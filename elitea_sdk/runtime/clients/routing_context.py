"""Bound runtime context for routing; no tool execution or source-content export."""
import hashlib
from functools import lru_cache


@lru_cache(maxsize=32)
def instruction_view(text):
    """Reuse CPU preparation only; classifier input remains billable each run."""
    limit = 8000
    return {'text': text if len(text) <= limit else text[:limit//2]+'\n[omitted middle]\n'+text[-limit//2:],
            'revision': hashlib.sha256(text.encode()).hexdigest(),
            'total_chars': len(text), 'truncated': len(text) > limit}


def retrieval_sources(tools, task):
    from ..middleware.project_context import ReadProjectContextTool
    from ..tools.vectorstore import VectorStoreWrapper
    from ..tools.tool_binding import _QualifiedToolAlias
    result = []
    for provider_tool in tools:
        tool = provider_tool.original_tool if isinstance(provider_tool, _QualifiedToolAlias) else provider_tool
        if isinstance(tool, ReadProjectContextTool):
            result.append({'source_id': 'project-context:'+tool.revision[:64],
                'tool_name': provider_tool.name, 'arguments': {}, 'description': tool.activation_description})
        elif isinstance(getattr(tool, 'api_wrapper', None), VectorStoreWrapper) and tool.name == 'search_documents':
            wrapper = tool.api_wrapper
            name = str(wrapper.vectorstore_params.get('collection_name') or '')
            if name:
                result.append({'source_id': 'vectorstore:'+hashlib.sha256(name.encode()).hexdigest(),
                    'tool_name': provider_tool.name, 'arguments': {'query': task[:500]},
                    'description': ('Search the configured document collection '+name)[:300]})
    return list({row['source_id']: row for row in result}.values())

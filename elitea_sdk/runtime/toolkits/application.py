import logging
from typing import List, Any, Optional

from langgraph.store.base import BaseStore
from pydantic import create_model, BaseModel, Field
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_core.tools import BaseTool
from ..tools.application import Application

logger = logging.getLogger(__name__)


def build_dynamic_application_schema(variables: list, app_name: str = "Application") -> type[BaseModel]:
    """
    Build a dynamic Pydantic schema for an Application tool that includes agent variables.

    This enables swarm-compatible agents where the orchestrator LLM can see and pass
    variables to child agents as tool parameters.

    Args:
        variables: List of variable dicts from version_details['variables'].
                  Each dict should have 'name' and optionally 'value', 'description'.
        app_name: Name of the application (used for schema naming)

    Returns:
        A dynamically created Pydantic model class with task and variable fields.
    """
    # Base fields - always present
    fields = {
        'task': (str, Field(description="Task for Application. Include all context needed by the application.")),
    }

    # Add agent variables as optional fields with their default values
    if variables:
        for var in variables:
            if not isinstance(var, dict) or not var.get('name'):
                continue

            var_name = var['name']
            var_description = var.get('description') or f"Variable: {var_name}"
            default_value = var.get('value')

            # Variables are optional strings with default values from the agent config
            # If default_value is None or empty string, we use None as default
            if default_value is None or default_value == '':
                fields[var_name] = (Optional[str], Field(
                    description=var_description,
                    default=None
                ))
            else:
                fields[var_name] = (Optional[str], Field(
                    description=var_description,
                    default=default_value
                ))

        logger.info(f"[APP_SCHEMA] Built dynamic schema for '{app_name}' with {len(variables)} variables")

    # Create a unique model name based on the application name
    # Clean the name to be a valid Python identifier
    safe_name = ''.join(c if c.isalnum() else '_' for c in app_name)
    model_name = f"{safe_name}Schema"

    return create_model(model_name, **fields)

def _build_application_description(
    base_description: Optional[str],
    tools: list,
) -> Optional[str]:
    """Build an enriched description for an Application tool.

    Appends a structured capabilities list derived from the nested agent's configured
    toolkits so the orchestrator LLM knows what the agent can handle and which individual
    tools are available. Enrichment is based purely on the agent's static configuration
    at bind time.
    """
    if not tools:
        return base_description

    _seen_labels = set()
    _capability_lines = []
    for tool in tools:
        _type = str(tool.get('type') or '')
        _label = (
            tool.get('toolkit_name')
            or tool.get('name')
            or _type
        )
        if not _label or _label in _seen_labels:
            continue
        _seen_labels.add(_label)
        _settings = tool.get('settings') or {}
        _selected = _settings.get('selected_tools') or []
        if _selected:
            _capability_lines.append(f"{_label}: {', '.join(_selected)}")
        else:
            _capability_lines.append(_label)

    if not _capability_lines:
        return base_description

    parts = []
    if base_description:
        parts.append(base_description.rstrip())
    parts.append("Configured capabilities:\n" + "\n".join(f"- {c}" for c in _capability_lines))

    return "\n".join(parts)


class ApplicationToolkit(BaseToolkit):
    tools: List[BaseTool] = []
    
    @staticmethod
    def toolkit_config_schema() -> BaseModel:
        return create_model(
            "application",
            # client = (Any, FieldInfo(description="Client object", required=True, autopopulate=True)),

            application_id = (int, Field(description="Application id")),
            application_version_id = (int, Field(description="Application version id")),
            app_api_key = (Optional[str], Field(description="Application API Key", autopopulate=True, default=None))
        )
    
    @classmethod
    def get_toolkit(cls, client: 'EliteAClient', application_id: int, application_version_id: int,
                    selected_tools: list[str] = [], store: Optional[BaseStore] = None,
                    ignored_mcp_servers: Optional[list] = None, is_subgraph: bool = False,
                    mcp_tokens: Optional[dict] = None, project_id: int = None,
                    conversation_id: Optional[str] = None, agent_type: str = 'agent',
                    memory: Optional[Any] = None,
                    fallback_llm=None,
                    user_declined_mcp_servers: Optional[list] = None):
        """
        Get toolkit for an application.

        Args:
            project_id: Optional project ID where the application lives.
                       If not specified, uses the client's default project.
                       This is needed for public project agents added as participants.
            agent_type: Type of agent ('agent', 'pipeline', 'predict') for metadata
        """
        logger.debug(f"[APP_TOOLKIT] get_toolkit called: app_id={application_id}, version_id={application_version_id}, "
                   f"project_id={project_id}, client.project_id={client.project_id}")

        # Check if accessing an application from a different project (public project)
        is_public_project = project_id is not None and project_id != client.project_id

        if is_public_project:
            # Use public application endpoint for cross-project access
            public_data = client.get_public_app_details(application_id)
            version_details = public_data.get('version_details', {})
            # Verify we got the expected version; re-fetch by version name if mismatched
            if version_details.get('id') != application_version_id:
                target_name = None
                for v in public_data.get('versions', []):
                    if v.get('id') == application_version_id:
                        target_name = v.get('name')
                        break
                if target_name:
                    public_data = client.get_public_app_details(application_id, version_name=target_name)
                    version_details = public_data.get('version_details', {})
            app_details = public_data
        else:
            # Use standard endpoints for same-project access
            app_details = client.get_app_details(application_id)
            version_details = client.get_app_version_details(application_id, application_version_id)

        # Resolve {{secret.xxx}} placeholders in MCP tool settings before passing to the SDK
        _tools = version_details.get('tools')
        if isinstance(_tools, list):
            import re as _re
            _secret_pat = _re.compile(r"\{\{secret\.([A-Za-z0-9_]+)\}\}")

            def _resolve_secrets(val):
                if isinstance(val, str):
                    def _sub(m):
                        try:
                            resolved = client.unsecret(m.group(1))
                            return resolved if resolved is not None else m.group(0)
                        except Exception:
                            return m.group(0)
                    return _secret_pat.sub(_sub, val)
                if isinstance(val, dict):
                    return {k: _resolve_secrets(v) for k, v in val.items()}
                if isinstance(val, list):
                    return [_resolve_secrets(v) for v in val]
                return val

            resolved_tools = []
            for _tool in _tools:
                if (isinstance(_tool, dict) and isinstance(_tool.get('type'), str)
                        and (_tool['type'] == 'mcp' or _tool['type'].startswith('mcp_'))
                        and 'settings' in _tool):
                    _tool = {**_tool, 'settings': _resolve_secrets(_tool['settings'])}
                resolved_tools.append(_tool)
            version_details = {**version_details, 'tools': resolved_tools}

        # Embedded sub-agents intentionally have null llm_settings; fall back to caller's LLM.
        llm_settings = version_details.get('llm_settings') or {}
        _model_name = llm_settings.get('model_name')
        if _model_name:
            model_settings = {
                "max_tokens": llm_settings.get('max_tokens'),
                "reasoning_effort": llm_settings.get('reasoning_effort'),
                "temperature": llm_settings.get('temperature'),
                # Honor this sub-agent's own OpenAI-passthrough routing. Without it the child
                # defaulted to ChatAnthropic, which drops tool_use blocks against an
                # OpenAI-passthrough backend (no tool calls -> no sensitive-tool HITL interrupts).
                "openai_compatible": llm_settings.get('openai_compatible', False),
            }
            resolved_llm = client.get_llm(_model_name, model_settings)
        else:
            model_settings = {}
            resolved_llm = fallback_llm

        app = client.application(application_id, application_version_id, store=store,
                                 llm=resolved_llm,
                                 memory=memory,
                                 ignored_mcp_servers=ignored_mcp_servers,
                                 is_subgraph=is_subgraph,
                                 mcp_tokens=mcp_tokens,
                                 conversation_id=conversation_id,
                                 version_details=version_details,
                                 user_declined_mcp_servers=user_declined_mcp_servers)  # Pass version_details to avoid re-fetching

        # Extract icon_meta from version_details meta field
        icon_meta = version_details.get('meta', {}).get('icon_meta', {})

        # Build dynamic args_schema that includes agent variables for swarm compatibility
        # This allows orchestrator LLMs to see and pass variables to child agents
        variables = version_details.get('variables', [])
        app_name = app_details.get("name", "Application")
        dynamic_schema = build_dynamic_application_schema(variables, app_name)

        # Extract variable defaults for use in Application._run()
        # This ensures default values are applied when variables are not explicitly passed
        variable_defaults = {}
        for var in variables:
            if isinstance(var, dict) and var.get('name'):
                default_val = var.get('value')
                if default_val is not None and default_val != '':
                    variable_defaults[var['name']] = default_val

        # Build metadata with toolkit_type, agent_type, and display name for chip rendering.
        # app_name is the human-readable agent/pipeline name shown on the chip label.
        metadata = {
            'toolkit_type': 'application',
            'agent_type': agent_type,
            'toolkit_name': app_name,   # used by FE resolveToolkitType() and chip label fallback
            'display_name': app_name,   # canonical display field consumed directly by chip label
        }
        if icon_meta:
            metadata['icon_meta'] = icon_meta

        # Build an enriched description so the parent LLM knows what capabilities
        # the nested agent actually has. Without this, the parent blindly delegates tasks
        # to the nested agent even when the required toolkit was skipped or unavailable.
        description = _build_application_description(
            app_details.get("description"),
            version_details.get('tools', []),
        )

        return cls(tools=[Application(name=app_name,
                                      description=description,
                                      application=app,
                                      args_schema=dynamic_schema,
                                      return_type='str',
                                      client=client,
                                      metadata=metadata,
                                      is_subgraph=is_subgraph,
                                      variable_defaults=variable_defaults,  # Store defaults for _run()
                                      args_runnable={
                                          "application_id": application_id,
                                          "application_version_id": application_version_id,
                                          "store": store,
                                          "llm": resolved_llm,
                                          "memory": memory,
                                          "ignored_mcp_servers": ignored_mcp_servers,
                                          "is_subgraph": is_subgraph,  # Pass is_subgraph flag
                                          "mcp_tokens": mcp_tokens,
                                          "conversation_id": conversation_id,
                                          "version_details": version_details,  # Include to avoid re-fetching (critical for public project apps)
                                          "user_declined_mcp_servers": user_declined_mcp_servers,
                                      })])
            
    def get_tools(self):
        return self.tools
    
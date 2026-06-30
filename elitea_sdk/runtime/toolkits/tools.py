import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from langchain_core.tools import StructuredTool, ToolException
from langgraph.store.base import BaseStore
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from elitea_sdk.tools import get_toolkits as elitea_toolkits
from elitea_sdk.tools import get_tools as elitea_tools
from .application import ApplicationToolkit
from .artifact import ArtifactToolkit
from .vectorstore import VectorStoreToolkit
from .mcp import McpToolkit
from .mcp_config import McpConfigToolkit, get_mcp_config_toolkit_schemas, get_mcp_server_config, load_mcp_servers_config
from ..tools.mcp_server_tool import McpServerTool
from ..tools.sandbox import SandboxToolkit
from ..tools.data_analysis import DataAnalysisToolkit
# Import community tools
from ...community import get_toolkits as community_toolkits, get_tools as community_tools
from ...tools.memory import MemoryToolkit
from ..utils.mcp_oauth import (
    canonical_resource,
    McpAuthorizationRequired,
    build_mcp_auth_decision_result,
    _is_http_url,
    mcp_alternate_resource,
    has_active_mcp_token,
    normalize_mcp_url,
)
from ...tools.utils import clean_string
from ..utils.utils import safe_config_summary
from elitea_sdk.tools import _inject_toolkit_id, _inject_display_metadata, _patch_tool_invoke

# Human-readable display names for all internal tools.
# Labels mirror INTERNAL_TOOLS_LIST[*].title in the FE to stay in sync.
# Tools that produce BaseTool objects (pyodide, data_analysis) use this for chip injection.
# The remaining entries serve as a complete registry and backwards-compat fallback.
INTERNAL_TOOL_DISPLAY_NAMES: dict = {
    'attachments':     'Attachments',           # artifact bucket; no chip event
    'image_generation': 'Image creation',       # provider toolkit; no chip event
    'data_analysis':   'Data Analysis',         # DataAnalysisToolkit — chip injected
    'planner':         'Planner',               # deprecated no-op; no chip event
    'pyodide':         'Python sandbox',        # SandboxToolkit — chip injected
    'swarm':           'Swarm Mode',            # mode flag; no chip event
    'lazy_tools_mode': 'Smart Tools Selection', # mode flag; no chip event
}
from .security import is_toolkit_blocked, is_tool_blocked, get_blocked_tools_for_toolkit


logger = logging.getLogger(__name__)


class _DeferredMcpAuthInput(PydanticBaseModel):
    # Keep schema permissive since we may not have full tool args before auth/discovery.
    arguments: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Arguments for the intended MCP tool call (ignored until authorization is completed).",
    )


def _safe_tool_name(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", str(value or "")).strip("_")
    return sanitized or "mcp_authorize"



def _infer_server_name(tool: dict, settings: dict) -> str:
    server_name = settings.get("server_name") or ""
    if not server_name and str(tool.get("type", "")).startswith("mcp_") and tool.get("type") != "mcp_config":
        server_name = str(tool.get("type", ""))[4:]
    return server_name


def _infer_proxy_tool_names(tool: dict, settings: dict) -> List[str]:
    # Always one proxy stub per unauthenticated server.
    # Expanding to N stubs (one per selected_tool) blows past the LLM tool limit and is
    # unnecessary — every stub returns the same authorization_required message directing
    # the LLM to call mcp_auth_control. One stub per server is sufficient.
    settings = tool.get("settings") or {}
    fallback_label = (
        tool.get("toolkit_name")
        or settings.get("server_name")
        or _infer_server_name(tool, settings)
        or settings.get("url")
        or "server"
    )
    return [f"mcp_authorize_{_safe_tool_name(str(fallback_label))}"]


def _build_deferred_mcp_auth_tools(
    tool: dict,
    auth_err: McpAuthorizationRequired,
    mcp_tokens: Optional[dict] = None,
    user_declined_mcp_servers: Optional[list] = None,
    force_declined: bool = False,
) -> List[StructuredTool]:
    """Build proxy tools that trigger MCP auth only when actually invoked.

    This prevents eager per-server auth prompts during initial tool loading when
    multiple MCP toolkits are unauthenticated.
    """
    settings = dict(tool.get("settings") or {})
    toolkit_type = str(tool.get("type") or auth_err.toolkit_type or "mcp")
    toolkit_name = str(tool.get("toolkit_name") or _infer_server_name(tool, settings) or toolkit_type)
    proxy_names = _infer_proxy_tool_names(tool, settings)

    server_url = auth_err.server_url
    if not _is_http_url(server_url):
        configured_url = None
        for _key in ("url", "server_url", "base_url", "authorization_server_url", "auth_url"):
            _candidate = settings.get(_key)
            if _is_http_url(_candidate):
                configured_url = _candidate
                break
        if not _is_http_url(configured_url):
            server_name = settings.get("server_name") or _infer_server_name(tool, settings)
            if not server_name and not _is_http_url(server_url):
                server_name = server_url  # treat the symbolic name itself as server_name
            if server_name:
                config = get_mcp_server_config(server_name) or {}
                # Case-insensitive fallback
                if not config:
                    _all = load_mcp_servers_config()
                    _lower = server_name.lower()
                    for _k, _v in _all.items():
                        if _k.lower() == _lower:
                            config = _v
                            break
                for _url_key in ("url", "server_url", "base_url", "authorization_server_url", "auth_url"):
                    _candidate = config.get(_url_key)
                    if _is_http_url(_candidate):
                        configured_url = _candidate
                        break
        if _is_http_url(configured_url):
            server_url = configured_url

    normalized_server_url = canonical_resource(server_url) if _is_http_url(server_url) else server_url
    resource_metadata_url = auth_err.resource_metadata_url
    www_authenticate = auth_err.www_authenticate
    resource_metadata = auth_err.resource_metadata

    # Check if the user has already declined auth for this server in this session.
    # If so, the proxy should tell the LLM the server is unavailable rather than
    # re-triggering the authorization flow.
    _server_is_declined = force_declined
    if not _server_is_declined and user_declined_mcp_servers and _is_http_url(normalized_server_url):
        for _dec in user_declined_mcp_servers:
            _dec_url = (
                _dec.get('server_url') or ''
                if isinstance(_dec, dict) else str(_dec or '')
            )
            if _dec_url and _is_http_url(_dec_url):
                if normalized_server_url == canonical_resource(normalize_mcp_url(_dec_url)):
                    _server_is_declined = True
                    break

    proxies: List[StructuredTool] = []
    for proxy_name in proxy_names:
        resolved_name = proxy_name

        def _deferred_mcp_tool(
            arguments: Optional[Dict[str, Any]] = None,
            _tool_name: str = resolved_name,
            _declined: bool = _server_is_declined,
        ) -> str:
            _ = arguments
            if _declined:
                # User already skipped auth for this server — do not re-trigger the flow.
                return build_mcp_auth_decision_result(
                    status="declined",
                    server_url=normalized_server_url,
                    tool_name=_tool_name,
                    toolkit_type=toolkit_type,
                    message=(
                        f"The '{toolkit_name}' MCP server was skipped for this run and is unavailable. "
                        "Do NOT request authorization, offer credentials, or suggest curl commands. "
                        "If other tools are available to complete or partially complete the task, use them. "
                        "If not, respond with one concise sentence explaining the task could not be completed "
                        f"because the {toolkit_name} server was skipped."
                    ),
                    next_step="use_other_tools_or_report",
                )
            # This proxy was created because toolkit loading already failed with
            # McpAuthorizationRequired — any token in mcp_tokens was rejected by
            # the server (expired or invalid). Do not short-circuit with a token
            # presence check; always instruct the LLM to call mcp_auth_control.
            return build_mcp_auth_decision_result(
                status="authorization_required",
                server_url=normalized_server_url,
                tool_name=_tool_name,
                toolkit_type=toolkit_type,
                message=(
                    "This MCP capability requires authorization. "
                    "You MUST call mcp_auth_control with action='authorize' and "
                    f"server_url='{normalized_server_url}' to start the authorization flow. "
                    "Do not respond to the user until authorization is triggered."
                ),
                next_step="authorize",
                resource_metadata_url=resource_metadata_url,
                www_authenticate=www_authenticate,
                resource_metadata=resource_metadata,
            )

        if _server_is_declined:
            _proxy_description = (
                f"Gateway for '{toolkit_name}' MCP server operations. "
                f"ALWAYS call this tool when any task involves '{toolkit_name}' before responding to the user. "
                "The tool result will tell you exactly what to do next."
            )
        else:
            _proxy_description = (
                f"Access gateway for '{toolkit_name}' MCP server tools. "
                f"The '{toolkit_name}' server requires authorization before its tools can be used. "
                f"When the user requests any '{toolkit_name}' operation, call this tool immediately "
                "to initiate the authorization flow — do NOT tell the user these tools are unavailable. "
                "After calling this tool, you MUST follow up by calling "
                "mcp_auth_control(action='authorize', server_url=...) to complete authorization."
            )
        proxies.append(
            StructuredTool.from_function(
                func=_deferred_mcp_tool,
                name=resolved_name,
                description=_proxy_description,
                args_schema=_DeferredMcpAuthInput,
                handle_tool_error=False,
                metadata={
                    "toolkit_name": toolkit_name,
                    "toolkit_type": toolkit_type,
                    "tool_name": resolved_name,
                },
            )
        )

    return proxies


def _make_mcp_auth_control_tool(
    tool_configs: list,
    mcp_tokens: Optional[dict] = None,
    user_declined_mcp_servers: Optional[list] = None,
    ignored_mcp_servers: Optional[list] = None,
) -> List[StructuredTool]:
    """Create the mcp_auth_control StructuredTool (and its legacy alias) for the predict path.

    In the application path these tools are created by indexer_agent._make_mcp_auth_tools()
    and injected via client.application(tools=...).  The predict path has no such injection
    point, so we create a functionally equivalent set here inside get_tools() the first time
    a deferred-auth proxy is generated.

    The richer version created by indexer_agent (with declined_servers context) takes
    precedence via deduplication in LangChainAssistant.__init__.
    """
    # Build declined-server metadata map keyed by canonical URL so re-prompt loop
    # prevention works correctly when the LLM retries mcp_auth_control after Skip.
    declined = user_declined_mcp_servers or []
    server_metadata: Dict[str, Dict[str, Any]] = {}
    for _s in declined:
        if not isinstance(_s, dict) or not _s.get("server_url"):
            continue
        _raw = _s["server_url"]
        _key = canonical_resource(_raw) if _is_http_url(_raw) else _raw
        server_metadata[_key] = {**_s, "server_url": _key}

    def _decline_reason(meta: Dict[str, Any]) -> str:
        reason = str(meta.get("skip_reason") or meta.get("denial_reason") or "").strip()
        return reason or "user skipped MCP login for this run"

    class _McpAuthControlInput(PydanticBaseModel):
        action: str = Field(default="authorize", description="Action to perform: 'authorize' to start OAuth flow, 'status' to check declined state, 'explain_skip' to generate structured skip guidance")
        server_url: Optional[str] = Field(default=None, description="MCP server URL that needs authorization (required for 'authorize'; optional for 'status'/'explain_skip')")
        tool_name: Optional[str] = Field(default=None, description="Name of the MCP tool that triggered auth")
        reason: Optional[str] = Field(default=None, description="Optional reason for authorization")

    class _RequestMcpAuthInput(PydanticBaseModel):
        server_url: str = Field(description="MCP server URL that needs authorization")

    def _mcp_auth_control(
        action: str = "authorize",
        server_url: str = "",
        tool_name: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> str:
        normalized_action = (action or "authorize").strip().lower()
        normalized_url = canonical_resource(server_url) if _is_http_url(server_url) else (server_url or "")

        # Resolve symbolic server name (e.g. "atlassian3") to an HTTP URL.
        # "atlassian3" is the toolkit's user-facing label, not the MCP config key.
        # Step 1: find the toolkit in tool_configs by name/toolkit_name, extract server_name from settings.
        # Step 2: look up that server_name in the MCP servers config to get the HTTP URL.
        if not _is_http_url(normalized_url) and normalized_url:
            # Step 1: find matching toolkit config to get the real server_name
            _lookup_name = normalized_url
            _lower_lookup = normalized_url.lower()
            for _tc in tool_configs:
                _tc_name = str(_tc.get("toolkit_name") or _tc.get("name") or "")
                if _tc_name.lower() == _lower_lookup:
                    _sn = (_tc.get("settings") or {}).get("server_name")
                    if _sn:
                        _lookup_name = _sn
                    break
            # Step 2: look up in MCP servers config (case-insensitive)
            _all_configs = load_mcp_servers_config()
            _server_cfg = _all_configs.get(_lookup_name) or {}
            if not _server_cfg:
                _lower = _lookup_name.lower()
                for _k, _v in _all_configs.items():
                    if _k.lower() == _lower:
                        _server_cfg = _v
                        break
            for _url_key in ("url", "server_url", "base_url", "authorization_server_url", "auth_url"):
                _candidate = _server_cfg.get(_url_key)
                if _is_http_url(_candidate):
                    normalized_url = canonical_resource(_candidate)
                    break

        if normalized_action not in ("authorize", "status", "explain_skip"):
            return build_mcp_auth_decision_result(
                status="error",
                server_url=normalized_url,
                tool_name=tool_name or "",
                message=f"Unknown action '{action}'. Supported actions: 'authorize', 'status', 'explain_skip'.",
                next_step="respond_without_tool",
            )

        if normalized_action == "status":
            _meta = server_metadata.get(normalized_url, {})
            if normalized_url in server_metadata:
                _decline = _decline_reason(_meta)
                return build_mcp_auth_decision_result(
                    status="declined",
                    server_url=normalized_url,
                    tool_name=_meta.get("tool_name") or tool_name or "",
                    message=(
                        "This MCP server was already skipped in this conversation. "
                        f"Skip reason: {_decline}. "
                        "Do NOT request authorization again. "
                        "Continue the task without this capability, or explain to the user "
                        "why the task cannot be completed."
                    ),
                    next_step="use_other_tool",
                    denial_reason=_decline,
                )
            return build_mcp_auth_decision_result(
                status="not_needed",
                server_url=normalized_url,
                tool_name=tool_name or "",
                message="No declined MCP auth state is tracked for this server in the current conversation.",
                next_step="respond_without_tool",
            )

        if normalized_action == "explain_skip":
            _meta = server_metadata.get(normalized_url, {})
            _decline = _decline_reason(_meta) if normalized_url in server_metadata else (reason or "user skipped MCP login for this run")
            return build_mcp_auth_decision_result(
                status="skipped",
                server_url=normalized_url,
                tool_name=tool_name or "",
                message=(
                    "Authorization was declined for THIS invocation. Do not retry the same blocked call now. "
                    "Continue with other available tools. If no alternative remains and the task fails, "
                    f"explicitly mention this skip reason: {_decline}."
                ),
                next_step="use_other_tool",
                denial_reason=_decline,
            )

        # authorize action
        if not normalized_url:
            return build_mcp_auth_decision_result(
                status="error",
                server_url="",
                tool_name=tool_name or "",
                message="server_url is required for mcp_auth_control.",
                next_step="respond_without_tool",
            )

        # Early return if the server was already declined in this conversation.
        # Do NOT call discover_mcp_tools — that would raise McpAuthorizationRequired
        # again and trigger another auth dialog even though the user already skipped.
        # Check canonical URL and Atlassian alternate URL to cover all token storage key forms.
        _atlassian_alt_url = mcp_alternate_resource(normalized_url) if normalized_url else None
        _declined_key = None
        for _check_key in [normalized_url, _atlassian_alt_url]:
            if _check_key and _check_key in server_metadata:
                _declined_key = _check_key
                break
        if _declined_key:
            _meta = server_metadata[_declined_key]
            _decline = _decline_reason(_meta)
            return build_mcp_auth_decision_result(
                status="declined",
                server_url=normalized_url,
                tool_name=_meta.get("tool_name") or tool_name or "",
                message=(
                    "This MCP server was already skipped in this conversation. "
                    f"Skip reason: {_decline}. "
                    "Do NOT request authorization again. "
                    "Continue the task without this capability, or explain to the user "
                    "why the task cannot be completed."
                ),
                next_step="use_other_tool",
                denial_reason=_decline,
            )

        # Check if this server was ignored (user clicked Skip) for this run.
        # Return declined immediately — do NOT call discover_mcp_tools, which would
        # raise McpAuthorizationRequired again and re-trigger the auth dialog.
        _ignored = ignored_mcp_servers or []
        if _ignored and normalized_url:
            _atlassian_alt_url_ignored = mcp_alternate_resource(normalized_url) if normalized_url else None
            for _check in [normalized_url, server_url, _atlassian_alt_url_ignored]:
                if _check and (_check in _ignored or canonical_resource(_check) in _ignored):
                    return build_mcp_auth_decision_result(
                        status="declined",
                        server_url=normalized_url,
                        tool_name=tool_name or "",
                        message=(
                            "This MCP server was skipped for this run and is unavailable. "
                            "Do NOT request authorization, offer credentials, or suggest curl commands. "
                            "If other tools are available to complete or partially complete the task, use them. "
                            "If not, respond with one concise sentence explaining the task could not be completed "
                            "because this MCP server was skipped."
                        ),
                        next_step="use_other_tools_or_report",
                        denial_reason="user skipped MCP login for this run",
                    )

        # Look up token from mcp_tokens (canonical key first, then raw URL, then Atlassian alternate)
        auth_headers: Dict[str, str] = {}
        token_session_id: Optional[str] = None
        if mcp_tokens:
            _atlassian_alt = mcp_alternate_resource(normalized_url) if normalized_url else None
            for lookup_key in [normalized_url, server_url, _atlassian_alt]:
                if not lookup_key:
                    continue
                token_data = mcp_tokens.get(lookup_key)
                if token_data:
                    if isinstance(token_data, dict):
                        access_token = token_data.get("access_token") or token_data.get("token")
                        token_session_id = token_data.get("session_id")
                    else:
                        access_token = str(token_data)
                    if access_token:
                        auth_headers["Authorization"] = f"Bearer {access_token}"
                    break

        # Attempt discovery; raises McpAuthorizationRequired if token is missing/invalid
        try:
            from ..utils.mcp_tools_discovery import discover_mcp_tools  # pylint: disable=C0415
            discover_mcp_tools(
                url=normalized_url,
                headers=auth_headers or None,
                timeout=30,
                session_id=token_session_id,
                ssl_verify=True,
            )
        except McpAuthorizationRequired as exc:
            if not getattr(exc, "server_url", None):
                exc.server_url = normalized_url
            if not getattr(exc, "tool_name", None):
                exc.tool_name = tool_name or ""
            raise
        except Exception as exc:
            logger.warning("[MCP Auth] mcp_auth_control discovery failed for %s: %s", normalized_url, exc)
            return build_mcp_auth_decision_result(
                status="error",
                server_url=normalized_url,
                tool_name=tool_name or "",
                message=f"Authorization check failed: {exc}",
                next_step="respond_without_tool",
            )

        if auth_headers.get("Authorization"):
            return build_mcp_auth_decision_result(
                status="authorized",
                server_url=normalized_url,
                tool_name=tool_name or "",
                message="Authorization successful. You may now use MCP tools for this server.",
                next_step="use_mcp_tool",
            )
        return build_mcp_auth_decision_result(
            status="not_needed",
            server_url=normalized_url,
            tool_name=tool_name or "",
            message="No authorization required for this server.",
            next_step="respond_without_tool",
        )

    def _request_mcp_authorization(server_url: str = "") -> str:
        return _mcp_auth_control(action="authorize", server_url=server_url)

    server_list = "\n".join(
        f"- {s.get('server_url', '')} reason: {_decline_reason(s)}"
        for s in declined
    )

    mcp_auth_control_tool = StructuredTool.from_function(
        func=_mcp_auth_control,
        name="mcp_auth_control",
        description=(
            "Control MCP authorization flow with structured decisions. "
            "Use action='authorize' when an MCP capability is required and user auth is needed. "
            "Use action='status' to check declined state in this conversation. "
            "Use action='explain_skip' to generate structured skip guidance and continue with alternatives.\n"
            "This is NOT a stop signal by itself: if auth is not granted, continue with other tools when possible.\n"
            "If a required MCP capability was skipped and no alternative works, the assistant response must "
            "explicitly state the skip reason.\n"
            f"Known declined servers in this conversation:\n{server_list or '- (none)'}"
        ),
        args_schema=_McpAuthControlInput,
        handle_tool_error=False,
    )

    legacy_alias_tool = StructuredTool.from_function(
        func=_request_mcp_authorization,
        name="request_mcp_authorization",
        description=(
            "Legacy alias for mcp_auth_control. "
            "Prefer calling mcp_auth_control(action='authorize', server_url=...) directly."
        ),
        args_schema=_RequestMcpAuthInput,
        handle_tool_error=False,
    )

    return [mcp_auth_control_tool, legacy_alias_tool]


def get_toolkits():
    # Note: Planning is now provided via PlanningMiddleware, not as a toolkit
    # See elitea_sdk.runtime.middleware.planning
    core_toolkits = [
        ArtifactToolkit.toolkit_config_schema(),
        MemoryToolkit.toolkit_config_schema(),
        VectorStoreToolkit.toolkit_config_schema(),
        SandboxToolkit.toolkit_config_schema(),
        DataAnalysisToolkit.toolkit_config_schema(),
        McpToolkit.toolkit_config_schema(),
        McpConfigToolkit.toolkit_config_schema(),
    ]

    # Add configured MCP servers (stdio and http) as available toolkits
    mcp_config_toolkits = get_mcp_config_toolkit_schemas()

    return core_toolkits + mcp_config_toolkits + community_toolkits() + elitea_toolkits()


def get_tools(tools_list: list, elitea_client=None, llm=None, memory_store: BaseStore = None, debug_mode: Optional[bool] = False, mcp_tokens: Optional[dict] = None, conversation_id: Optional[str] = None, ignored_mcp_servers: Optional[list] = None, current_participant_id: Optional[int] = None, memory: Optional[object] = None, user_declined_mcp_servers: Optional[list] = None, pipeline_node_toolkit_names: Optional[set] = None, skipped_pipeline_toolkit_names: Optional[set] = None) -> list:
    """
    Process tool configurations and return instantiated tools.

    Args:
        current_participant_id: The participant ID of the agent being predicted to.
            Used to filter out self-references (prevent agent from calling itself).
    """
    # Sanitize tools_list to handle corrupted tool configurations
    sanitized_tools = []
    seen_toolkit_ids = set()  # Track seen toolkit IDs for deduplication

    for tool in tools_list:
        if isinstance(tool, dict):
            # Check for corrupted structure where 'type' and 'name' contain the full tool config
            if 'type' in tool and isinstance(tool['type'], dict):
                # This is a corrupted tool - use the inner dict instead
                logger.warning(f"Detected corrupted tool configuration (type=dict), fixing: {safe_config_summary(tool)}")
                actual_tool = tool['type']  # or tool['name'], they should be the same
                sanitized_tools.append(actual_tool)
            elif 'name' in tool and isinstance(tool['name'], dict):
                # Another corruption pattern where name contains the full config
                logger.warning(f"Detected corrupted tool configuration (name=dict), fixing: {safe_config_summary(tool)}")
                actual_tool = tool['name']
                sanitized_tools.append(actual_tool)
            elif 'type' in tool and isinstance(tool['type'], str):
                # Valid tool configuration
                sanitized_tools.append(tool)
            else:
                # Skip invalid/corrupted tools that can't be fixed
                logger.warning(f"Skipping invalid tool configuration: {safe_config_summary(tool)}")
        else:
            logger.warning(f"Skipping non-dict tool: {safe_config_summary(tool)}")
            # Skip non-dict tools

    # Deduplication and self-filtering
    deduplicated_tools = []
    for tool in sanitized_tools:
        # Deduplicate by toolkit ID (for toolkits that have an ID)
        toolkit_id = tool.get('id')
        if toolkit_id is not None:
            if toolkit_id in seen_toolkit_ids:
                logger.debug(f"Skipping duplicate toolkit id={toolkit_id}")
                continue
            seen_toolkit_ids.add(toolkit_id)

        # Self-filtering for application tools (prevent agent from calling itself)
        if tool.get('type') == 'application' and current_participant_id is not None:
            participant_id = tool.get('participant_id')
            if participant_id == current_participant_id:
                logger.info(f"Filtering out self-reference: participant_id={participant_id}")
                continue

        # Security filtering - block configured toolkits at runtime
        tool_type = tool.get('type', '')
        if is_toolkit_blocked(tool_type):
            logger.warning(f"[SECURITY] Skipping blocked toolkit type '{tool_type}' "
                          f"(toolkit_id={toolkit_id}, name={tool.get('name', 'unknown')})")
            continue

        deduplicated_tools.append(tool)

    tools = []
    unhandled_tools = []  # Track tools not handled by main processing
    _mcp_auth_control_added = False  # Ensure mcp_auth_control is injected at most once

    for tool in deduplicated_tools:
        # Flag to track if this tool was processed by the main loop
        # Used to prevent double processing by fallback systems
        tool_handled = False
        # # --- OAuth token injection for non-MCP toolkits (SharePoint and others) ---
        # try:
        #     settings_preview = tool.get('settings', {}) if isinstance(tool, dict) else {}
        #     # Determine common URL fields used by toolkits
        #     toolkit_url = settings_preview.get('base_url') or settings_preview.get('site_url') or settings_preview.get('url')
        #     session_id_from_token = None
        #     access_token = None
        #     if mcp_tokens and toolkit_url:
        #         try:
        #             canonical_url = canonical_resource(toolkit_url)
        #         except Exception:
        #             canonical_url = toolkit_url
        #         # Prefer canonical key, fallback to raw URL
        #         token_data = mcp_tokens.get(canonical_url) or mcp_tokens.get(toolkit_url)
        #         if token_data:
        #             if isinstance(token_data, dict):
        #                 access_token = token_data.get('access_token') or token_data.get('token') or None
        #                 session_id_from_token = token_data.get('session_id')
        #             else:
        #                 access_token = token_data
        #
        #     if access_token:
        #         # Inject token for SharePoint toolkit (expects `token` setting)
        #         if tool.get('type') == 'sharepoint' or 'site_url' in settings_preview:
        #             settings = dict(tool.get('settings', {}) or {})
        #             settings['token'] = access_token
        #             if session_id_from_token:
        #                 settings['session_id'] = session_id_from_token
        #             tool['settings'] = settings
        #             logger.info(f"[OAUTH] Injected SharePoint token for toolkit {tool.get('name')}")
        #         else:
        #             # Generic injection: set Authorization header in settings.headers
        #             settings = dict(tool.get('settings', {}) or {})
        #             headers = dict(settings.get('headers') or {})
        #             headers.setdefault('Authorization', f'Bearer {access_token}')
        #             settings['headers'] = headers
        #             if session_id_from_token:
        #                 settings['session_id'] = session_id_from_token
        #             tool['settings'] = settings
        #             logger.info(f"[OAUTH] Injected Authorization header for toolkit {tool.get('name')}")
        # except Exception:
        #     # Token injection must be non-fatal
        #     logger.debug("OAuth token injection skipped due to an error", exc_info=True)
        try:
            if tool['type'] == 'application':
                tool_handled = True
                is_application_subgraph = True
                # Get project_id from top level (injected by BE for all data paths)
                app_project_id = tool.get('project_id')
                # Get agent_type for metadata
                agent_type = tool.get('agent_type', 'agent')
                logger.info(f"[APP_TOOL] Processing application tool '{tool.get('name')}': "
                           f"app_id={tool['settings'].get('application_id')}, "
                           f"version_id={tool['settings'].get('application_version_id')}, "
                           f"project_id={app_project_id}, "
                           f"agent_type={agent_type}, "
                           f"raw_settings={tool.get('settings')}")

                try:
                    tools.extend(ApplicationToolkit.get_toolkit(
                        elitea_client,
                        application_id=int(tool['settings']['application_id']),
                        application_version_id=int(tool['settings']['application_version_id']),
                        selected_tools=[],
                        ignored_mcp_servers=ignored_mcp_servers,
                        is_subgraph=is_application_subgraph,
                        mcp_tokens=mcp_tokens,
                        project_id=app_project_id,  # Use agent's project, not conversation's
                        conversation_id=conversation_id,
                        agent_type=agent_type,  # Pass agent_type for metadata
                        memory=memory,
                        fallback_llm=llm,  # Fallback for embedded sub-agents with null llm_settings
                        user_declined_mcp_servers=user_declined_mcp_servers,
                    ).get_tools())
                except Exception as app_err:
                    # Gracefully skip application tools that fail to load (e.g., deleted agents,
                    # or participants whose nested MCP requires auth — Application._run() rebuilds
                    # the assistant on invocation, so init-time failures are non-fatal).
                    logger.error(f"Skipping application tool '{tool.get('name', 'unknown')}': {app_err}")
                    continue
            elif tool['type'] == 'memory':
                tool_handled = True
                memory_tools = MemoryToolkit.get_toolkit(
                    namespace=tool['settings'].get('namespace', str(tool['id'])),
                    pgvector_configuration=tool['settings'].get('pgvector_configuration', {}),
                    store=memory_store,
                    toolkit_name=tool.get('name', ''),
                ).get_tools()
                _inject_display_metadata(tool, memory_tools)
                tools += memory_tools
            # TODO: update configuration of internal tools
            elif tool['type'] == 'internal_tool':
                tool_handled = True
                internal_tools = []
                if tool['name'] == 'pyodide':
                    internal_tools = SandboxToolkit.get_toolkit(
                        stateful=False,
                        allow_net=True,
                        elitea_client=elitea_client,
                    ).get_tools()
                elif tool['name'] == 'planner':
                    # Planning is now provided via PlanningMiddleware, not as an internal tool
                    # See elitea_sdk.runtime.middleware.planning
                    logger.warning("'planner' internal tool is deprecated. Use PlanningMiddleware instead.")
                elif tool['name'] == 'data_analysis':
                    # Data Analysis internal tool - uses conversation attachment bucket
                    settings = tool.get('settings', {})
                    bucket_name = settings.get('bucket_name')
                    if bucket_name:
                        internal_tools = DataAnalysisToolkit.get_toolkit(
                            elitea_client=elitea_client,
                            llm=llm,
                            bucket_name=bucket_name,
                            toolkit_name="Data Analyst",
                        ).get_tools()
                    else:
                        logger.warning("Data Analysis internal tool requested "
                                       "but no bucket_name provided in settings")
                # Inject display metadata so FE chips show human-readable names
                if internal_tools:
                    internal_display_name = INTERNAL_TOOL_DISPLAY_NAMES.get(
                        tool['name'], tool['name']
                    )
                    for t in internal_tools:
                        if not hasattr(t, 'metadata'):
                            continue
                        if t.metadata is None:
                            t.metadata = {}
                        if isinstance(t.metadata, dict):
                            # Preserve toolkit_type if the tool already defines one
                            # (e.g. sandbox tools use 'sandbox' to match sensitive-tools config)
                            if 'toolkit_type' not in t.metadata:
                                t.metadata['toolkit_type'] = 'internal'
                            t.metadata['toolkit_name'] = tool['name']           # raw code name; fallback key
                            t.metadata['display_name'] = internal_display_name  # human-readable; chip label
                            _patch_tool_invoke(t)  # forward metadata into LangGraph run config
                tools.extend(internal_tools)
            elif tool['type'] == 'artifact':
                tool_handled = True
                toolkit_tools = ArtifactToolkit.get_toolkit(
                    client=elitea_client,
                    bucket=tool['settings']['bucket'],
                    toolkit_name=tool.get('toolkit_name', ''),
                    selected_tools=tool['settings'].get('selected_tools', []),
                    llm=llm,
                    # indexer settings
                    pgvector_configuration=tool['settings'].get('pgvector_configuration', {}),
                    embedding_model=tool['settings'].get('embedding_model'),
                    collection_name=f"{tool.get('toolkit_name')}",
                    collection_schema=str(tool['settings'].get('id', tool.get('id', ''))),
                ).get_tools()
                # Inject toolkit_id for artifact tools as well
                # Pass settings as the tool config since that's where the id field is
                _inject_toolkit_id(tool['settings'], toolkit_tools)
                _inject_display_metadata(tool, toolkit_tools)
                tools.extend(toolkit_tools)

            elif tool['type'] == 'vectorstore':
                tool_handled = True
                vs_tools = VectorStoreToolkit.get_toolkit(
                    llm=llm,
                    toolkit_name=tool.get('toolkit_name', ''),
                    **tool['settings']).get_tools()
                _inject_display_metadata(tool, vs_tools)
                tools.extend(vs_tools)
            elif tool['type'] == 'planning':
                tool_handled = True
                # Planning is now provided via PlanningMiddleware, not as a toolkit type
                # See elitea_sdk.runtime.middleware.planning
                logger.warning("'planning' toolkit type is deprecated. Use PlanningMiddleware instead.")
            elif tool['type'] == 'mcp':
                tool_handled = True
                # remote mcp tool initialization with token injection
                settings = dict(tool['settings'])
                url = settings.get('url')

                # Normalize deprecated endpoint forms (e.g. Atlassian /v1/sse -> /v1/mcp/authv2)
                if url:
                    normalized_url = normalize_mcp_url(url)
                    if normalized_url != url:
                        logger.info("[MCP] Normalizing deprecated endpoint to current form")
                        url = normalized_url
                        settings['url'] = url

                # Check if this MCP server should be ignored (user chose to continue without auth)
                # or was explicitly declined (user clicked Skip on the auth dialog).
                if url:
                    canonical_url = canonical_resource(url)
                    _should_skip = bool(
                        ignored_mcp_servers
                        and (canonical_url in ignored_mcp_servers or url in ignored_mcp_servers)
                    )
                    if not _should_skip and user_declined_mcp_servers:
                        for _dec in user_declined_mcp_servers:
                            _dec_url = (
                                _dec.get('server_url') or ''
                                if isinstance(_dec, dict) else str(_dec or '')
                            )
                            if _dec_url and _is_http_url(_dec_url):
                                if canonical_url == canonical_resource(normalize_mcp_url(_dec_url)):
                                    _should_skip = True
                                    break
                    if _should_skip:
                        logger.info("[MCP Auth] Skipping ignored/declined MCP server — injecting declined proxy")
                        _tname = tool.get('toolkit_name') or url
                        if (
                            skipped_pipeline_toolkit_names is not None
                            and pipeline_node_toolkit_names is not None
                            and _tname in pipeline_node_toolkit_names
                        ):
                            skipped_pipeline_toolkit_names.add(_tname)
                        _fake_auth_err = McpAuthorizationRequired(
                            message="MCP server skipped by user",
                            server_url=canonical_url or url,
                        )
                        _fake_auth_err.toolkit_type = tool.get('type', 'mcp')
                        _declined_proxies = _build_deferred_mcp_auth_tools(
                            tool, _fake_auth_err, mcp_tokens=mcp_tokens, force_declined=True
                        )
                        if _declined_proxies:
                            _inject_display_metadata(tool, _declined_proxies)
                            tools.extend(_declined_proxies)
                            if not _mcp_auth_control_added:
                                tools.extend(_make_mcp_auth_control_tool(
                                    deduplicated_tools, mcp_tokens=mcp_tokens,
                                    user_declined_mcp_servers=user_declined_mcp_servers,
                                    ignored_mcp_servers=ignored_mcp_servers,
                                ))
                                _mcp_auth_control_added = True
                        continue
                
                headers = settings.get('headers')
                token_data = None
                session_id = None
                if mcp_tokens and url:
                    canonical_url = canonical_resource(url)
                    logger.debug("[MCP Auth] Looking up token for MCP server")
                    logger.debug("[MCP Auth] Token lookup — %d known servers", len(mcp_tokens))
                    lookup_candidates = [canonical_url, url]
                    atlassian_alt = mcp_alternate_resource(canonical_url)
                    if atlassian_alt:
                        lookup_candidates.append(atlassian_alt)
                    matched_candidates = [candidate for candidate in lookup_candidates if candidate in mcp_tokens]
                    token_data = mcp_tokens.get(canonical_url)
                    matched_key = canonical_url if token_data is not None else None
                    if token_data is None and matched_candidates:
                        matched_key = matched_candidates[0]
                        token_data = mcp_tokens.get(matched_key)
                    if token_data:
                        logger.debug("[MCP Auth] Found token data for matched server")
                        # Handle both old format (string) and new format (dict with access_token and session_id)
                        if isinstance(token_data, dict):
                            access_token = token_data.get('access_token')
                            session_id = token_data.get('session_id')
                            logger.debug("[MCP Auth] Token data: access_token=%s, session_id=%s",
                                         'present' if access_token else 'missing',
                                         'present' if session_id else 'none')
                        else:
                            # Backward compatibility: treat as plain token string
                            access_token = token_data
                            logger.debug("[MCP Auth] Using legacy token format (string)")
                    else:
                        access_token = None
                        logger.debug("[MCP Auth] No token found for this MCP server")
                else:
                    access_token = None
                    
                if access_token:
                    merged_headers = dict(headers) if headers else {}
                    merged_headers.setdefault('Authorization', f'Bearer {access_token}')
                    settings['headers'] = merged_headers
                    # If Authorization was NOT already in the DB-configured headers, the token
                    # we just injected came from the OAuth flow (mcp_tokens), not from a static
                    # credential.  Signal this so a 401 re-triggers OAuth instead of a ValueError.
                    if not any(k.lower() == 'authorization' for k in (headers or {})):
                        settings['_oauth_token_injected'] = True
                    logger.debug("[MCP Auth] Added Authorization header for MCP server")
                    
                # Pass session_id to MCP toolkit if available
                if session_id:
                    settings['session_id'] = session_id
                    logger.debug("[MCP Auth] Passing session_id to toolkit")
                try:
                    mcp_tools = McpToolkit.get_toolkit(
                        toolkit_name=tool.get('toolkit_name', ''),
                        client=elitea_client,
                        **settings).get_tools()
                except McpAuthorizationRequired as auth_err:
                    auth_err.toolkit_type = tool['type']
                    _is_pipeline_node = (
                        pipeline_node_toolkit_names is not None
                        and tool.get('toolkit_name', '') in pipeline_node_toolkit_names
                    )
                    if _is_pipeline_node:
                        # Pipeline nodes call tools directly — deferred stubs are useless.
                        # Re-raise so the caller can trigger the OAuth flow for the user.
                        logger.info(
                            "[MCP Auth] Pipeline node toolkit requires authorization — re-raising"
                        )
                        raise
                    mcp_tools = _build_deferred_mcp_auth_tools(tool, auth_err, mcp_tokens=mcp_tokens, user_declined_mcp_servers=user_declined_mcp_servers)
                    logger.info(
                        "[MCP Auth] Deferred authorization for toolkit with %d proxy tool(s)",
                        len(mcp_tools),
                    )
                    if mcp_tools and not _mcp_auth_control_added:
                        tools.extend(_make_mcp_auth_control_tool(deduplicated_tools, mcp_tokens=mcp_tokens, user_declined_mcp_servers=user_declined_mcp_servers, ignored_mcp_servers=ignored_mcp_servers))
                        _mcp_auth_control_added = True
                        logger.info("[MCP Auth] Injected mcp_auth_control into predict-path toolset (mcp type)")
                    _inject_display_metadata(tool, mcp_tools)
                    tools.extend(mcp_tools)
                else:
                    _inject_display_metadata(tool, mcp_tools)
                    tools.extend(mcp_tools)
            elif tool['type'] == 'mcp_config' or tool['type'].startswith('mcp_'):
                tool_handled = True
                # MCP Config toolkit - pre-configured MCP servers (stdio or http)
                # Handle both explicit 'mcp_config' type and dynamic names like 'mcp_playwright'
                logger.info(f"Processing mcp_config toolkit: {safe_config_summary(tool)}")
                try:
                    settings = tool.get('settings', {})

                    # Server name can come from settings or be extracted from type name
                    server_name = settings.get('server_name')
                    if not server_name and tool['type'].startswith('mcp_') and tool['type'] != 'mcp_config':
                        # Extract server name from type (e.g., 'mcp_playwright' -> 'playwright')
                        server_name = tool['type'][4:]  # Remove 'mcp_' prefix

                    if not server_name:
                        logger.error(f"❌ No server_name found for mcp_config toolkit: {safe_config_summary(tool)}")
                        continue

                    if ignored_mcp_servers or user_declined_mcp_servers:
                        # Check by server_name / toolkit type (stdio and legacy cases)
                        _skip = bool(
                            ignored_mcp_servers
                            and (tool['type'] in ignored_mcp_servers or server_name in ignored_mcp_servers)
                        )
                        # Resolve the HTTP URL for this server (needed for both ignored and declined checks)
                        _server_url = (
                            settings.get('url')
                            or (settings.get('server_config') or {}).get('url')
                        )
                        if not _server_url:
                            _global_cfg = get_mcp_server_config(server_name) or {}
                            _server_url = _global_cfg.get('url')
                        if _server_url:
                            _server_url = normalize_mcp_url(_server_url)
                        if not _skip and ignored_mcp_servers and _server_url:
                            # Also check by HTTP URL for http-type pre-configured servers.
                            _canonical = canonical_resource(_server_url)
                            _skip = _canonical in ignored_mcp_servers or _server_url in ignored_mcp_servers
                        if not _skip and user_declined_mcp_servers:
                            # Check by name (stdio servers) or by HTTP URL (http-type servers)
                            for _dec in user_declined_mcp_servers:
                                if not isinstance(_dec, dict):
                                    continue
                                _dec_url = _dec.get('server_url') or ''
                                if _dec_url and not _is_http_url(_dec_url):
                                    # Symbolic server name — match by name or toolkit type
                                    if _dec_url == server_name or _dec_url == tool['type']:
                                        _skip = True
                                        break
                                elif _dec_url and _is_http_url(_dec_url) and _server_url:
                                    if canonical_resource(_server_url) == canonical_resource(normalize_mcp_url(_dec_url)):
                                        _skip = True
                                        break
                        if _skip:
                            logger.info("[MCP Auth] Skipping ignored/declined pre-configured MCP server — injecting declined proxy")
                            _tname = tool.get('toolkit_name') or server_name
                            if (
                                skipped_pipeline_toolkit_names is not None
                                and pipeline_node_toolkit_names is not None
                                and _tname in pipeline_node_toolkit_names
                            ):
                                skipped_pipeline_toolkit_names.add(_tname)
                            _fake_skip_url = _server_url or server_name
                            _fake_auth_err = McpAuthorizationRequired(
                                message="MCP server skipped by user",
                                server_url=_fake_skip_url,
                            )
                            _fake_auth_err.toolkit_type = tool.get('type', 'mcp_config')
                            _declined_proxies = _build_deferred_mcp_auth_tools(
                                tool, _fake_auth_err, mcp_tokens=mcp_tokens, force_declined=True
                            )
                            if _declined_proxies:
                                _inject_display_metadata(tool, _declined_proxies)
                                tools.extend(_declined_proxies)
                                if not _mcp_auth_control_added:
                                    tools.extend(_make_mcp_auth_control_tool(
                                        deduplicated_tools, mcp_tokens=mcp_tokens,
                                        user_declined_mcp_servers=user_declined_mcp_servers,
                                        ignored_mcp_servers=ignored_mcp_servers,
                                    ))
                                    _mcp_auth_control_added = True
                            continue

                    toolkit_name = tool.get('toolkit_name', '') or server_name
                    selected_tools = settings.get('selected_tools', [])
                    excluded_tools = settings.get('excluded_tools', [])

                    # Get server config (may be in settings or from global config)
                    server_config = settings.get('server_config')
                    toolkit_tools = McpConfigToolkit.get_toolkit(
                        server_name=server_name,
                        server_config=server_config,
                        user_config=settings,
                        selected_tools=selected_tools if selected_tools else None,
                        excluded_tools=excluded_tools if excluded_tools else None,
                        toolkit_name=toolkit_name,
                        client=elitea_client,
                        mcp_tokens=mcp_tokens,
                    ).get_tools()

                    _inject_display_metadata(tool, toolkit_tools)
                    tools.extend(toolkit_tools)
                    logger.info(f"✅ Successfully added {len(toolkit_tools)} tools from McpConfigToolkit ({server_name})")
                except McpAuthorizationRequired as auth_err:
                    auth_err.toolkit_type = tool['type']
                    _is_pipeline_node = (
                        pipeline_node_toolkit_names is not None
                        and tool.get('toolkit_name', '') in pipeline_node_toolkit_names
                    )
                    if _is_pipeline_node:
                        logger.info(
                            "[MCP Auth] Pipeline node toolkit requires authorization — re-raising"
                        )
                        raise
                    toolkit_tools = _build_deferred_mcp_auth_tools(tool, auth_err, mcp_tokens=mcp_tokens, user_declined_mcp_servers=user_declined_mcp_servers)
                    _inject_display_metadata(tool, toolkit_tools)
                    tools.extend(toolkit_tools)
                    logger.info(
                        "[MCP Auth] Deferred authorization for pre-configured MCP with %d proxy tool(s)",
                        len(toolkit_tools),
                    )
                    if toolkit_tools and not _mcp_auth_control_added:
                        tools.extend(_make_mcp_auth_control_tool(deduplicated_tools, mcp_tokens=mcp_tokens, user_declined_mcp_servers=user_declined_mcp_servers, ignored_mcp_servers=ignored_mcp_servers))
                        _mcp_auth_control_added = True
                        logger.info("[MCP Auth] Injected mcp_auth_control into predict-path toolset (mcp_config type)")
                    continue
                except Exception as e:
                    logger.error(f"❌ Failed to initialize McpConfigToolkit: {e}")
                    if not debug_mode:
                        raise
        except McpAuthorizationRequired:
            # Re-raise auth required exceptions directly
            raise
        except Exception as e:
            logger.error(f"Error initializing toolkit for tool '{tool.get('name', 'unknown')}': {e}", exc_info=True)
            if debug_mode:
                logger.info("Skipping tool initialization error due to debug mode.")
                continue
            else:
                raise ToolException(f"Error initializing toolkit for tool '{tool.get('name', 'unknown')}': {e}")

        # Track unhandled tools (make a copy to avoid reference issues)
        if not tool_handled:
            # Ensure we only add valid tool configurations to unhandled_tools
            if isinstance(tool, dict) and 'type' in tool and isinstance(tool['type'], str):
                unhandled_tools.append(dict(tool))

    # Add community tools (only for unhandled tools)
    community_loaded = community_tools(unhandled_tools, elitea_client, llm)
    tools += community_loaded
    logger.debug(f"[RUNTIME_TOOLS] Community tools loaded: {len(community_loaded)} tools")

    # Add elitea tools (only for unhandled tools)
    # set tokens to tools in order to handle case when token is required for authentication
    # Tool must have its own logic of handling it
    if mcp_tokens:
        for tool in unhandled_tools:
            if 'settings' not in tool:
                tool['settings'] = {}
            tool['settings']['tokens'] = mcp_tokens
    elitea_loaded = elitea_tools(unhandled_tools, elitea_client, llm, memory_store)
    tools += elitea_loaded
    logger.debug(f"[RUNTIME_TOOLS] EliteA tools loaded: {len(elitea_loaded)} tools")

    # Add MCP tools registered via elitea-mcp CLI (static registry)
    # Note: Tools with type='mcp' are already handled in main loop above
    mcp_loaded = _mcp_tools(unhandled_tools, elitea_client)
    tools += mcp_loaded
    logger.debug(f"[RUNTIME_TOOLS] MCP tools loaded: {len(mcp_loaded)} tools")

    # Final logging of all tools being returned
    all_tool_names = [t.name if hasattr(t, 'name') else str(type(t)) for t in tools]
    logger.debug(f"[RUNTIME_TOOLS] Total tools being returned: {len(tools)}")
    logger.debug(f"[RUNTIME_TOOLS] All tool names: {all_tool_names}")

    # Defence-in-depth: final blocked-tool sweep across ALL tools regardless of source.
    # Earlier filters cover main-loop toolkits; this catches community / elitea / MCP tools.
    pre_filter_count = len(tools)
    tools = _final_blocked_tools_filter(tools)
    if len(tools) < pre_filter_count:
        logger.info(
            "[SECURITY] Final blocked-tool filter removed %d tool(s)",
            pre_filter_count - len(tools),
        )

    # Check for indexer tools in the final list
    indexer_tools_final = [n for n in all_tool_names if 'index' in n.lower()]
    if indexer_tools_final:
        logger.warning(f"[RUNTIME_TOOLS] FINAL TOOL LIST contains indexer tools: {indexer_tools_final}")

    # Sanitize tool names to meet OpenAI's function naming requirements
    # tools = _sanitize_tool_names(tools)

    return tools


def _final_blocked_tools_filter(tools: list) -> list:
    """Remove any remaining blocked tools from the final tool list.

    Each tool's metadata is inspected for ``toolkit_type`` so the check
    matches the same keys used by ``configure_blocklist``.
    """
    from langchain_core.tools import BaseTool

    filtered = []
    for tool in tools:
        if not isinstance(tool, BaseTool):
            filtered.append(tool)
            continue
        metadata = getattr(tool, 'metadata', None) or {}
        toolkit_type = (
            metadata.get('toolkit_type')
            or metadata.get('type')
            or ''
        )
        if toolkit_type and is_tool_blocked(toolkit_type, tool.name):
            logger.warning(
                "[SECURITY] Final filter: removing blocked tool '%s' (type '%s')",
                tool.name, toolkit_type,
            )
            continue
        filtered.append(tool)
    return filtered


def _sanitize_tool_names(tools: list) -> list:
    """
    Sanitize tool names to meet LLM provider function naming requirements.
    Tool names must match pattern ^[a-zA-Z0-9_-]{1,128}$
    """
    import re
    from langchain_core.tools import BaseTool
    
    def sanitize_name(name):
        """Sanitize a single tool name"""
        # Replace dots with underscores (dots not allowed in tool names)
        sanitized = name.replace('.', '_')
        # Replace spaces and other invalid characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', sanitized)
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_{2,}', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized
    
    sanitized_tools = []
    name_mapping = {}
    
    for tool in tools:
        if isinstance(tool, BaseTool):
            original_name = tool.name
            sanitized_name = sanitize_name(original_name)
            
            # Only update if the name actually changed
            if original_name != sanitized_name:
                logger.info(f"Sanitizing tool name: '{original_name}' -> '{sanitized_name}'")
                # Create a new tool instance with the sanitized name
                # We need to be careful here to preserve all other tool properties
                tool.name = sanitized_name
                name_mapping[original_name] = sanitized_name
            
            sanitized_tools.append(tool)
        else:
            # For non-BaseTool objects (like CompiledStateGraph), just pass through
            sanitized_tools.append(tool)
    
    if name_mapping:
        logger.info(f"Tool name sanitization complete. Mapped {len(name_mapping)} tool names.")
    
    return sanitized_tools


def _mcp_tools(tools_list, elitea):
    """
    Handle MCP tools registered via elitea-mcp CLI (static registry).
    Skips tools with type='mcp' as those are handled by dynamic discovery.
    """
    try:
        all_available_toolkits = elitea.get_mcp_toolkits()
        toolkit_lookup = {tk["name"]: tk for tk in all_available_toolkits}
        tools = []
        #
        for selected_toolkit in tools_list:
            server_toolkit_name = selected_toolkit['type']
            
            # Skip tools with type='mcp' - they're handled by dynamic discovery
            if server_toolkit_name == 'mcp':
                continue
            
            toolkit_conf = toolkit_lookup.get(server_toolkit_name)
            #
            if not toolkit_conf:
                logger.debug(f"Toolkit '{server_toolkit_name}' not found in available MCP toolkits. Skipping...")
                continue
            #
            available_tools = toolkit_conf.get("tools", [])
            selected_tools = [name.lower() for name in selected_toolkit['settings'].get('selected_tools', [])]
            for available_tool in available_tools:
                tool_name = available_tool.get("name", "").lower()
                if not selected_tools or tool_name in selected_tools:
                    if server_tool := _init_single_mcp_tool(server_toolkit_name,
                                                            # selected_toolkit["name"] is None for toolkit_test
                                                            selected_toolkit["toolkit_name"] if selected_toolkit.get("toolkit_name")
                                                            else server_toolkit_name,
                                                            available_tool, elitea, selected_toolkit['settings']):
                        tools.append(server_tool)
        return tools
    except Exception:
        logger.error("Error while fetching MCP tools", exc_info=True)
        return []


def _init_single_mcp_tool(server_toolkit_name, toolkit_name, available_tool, elitea, toolkit_settings):
    try:
        # Use clean tool name without prefix
        tool_name = available_tool["name"]
        # Add toolkit context to description (max 1000 chars)
        toolkit_context = f" [Toolkit: {clean_string(toolkit_name)}]" if toolkit_name else ''
        base_description = f"MCP for a tool '{tool_name}': {available_tool.get('description', '')}"
        description = base_description
        if toolkit_context and len(base_description + toolkit_context) <= 1000:
            description = base_description + toolkit_context
        
        tool = McpServerTool(
            name=tool_name,
            description=description,
            args_schema=McpServerTool.create_pydantic_model_from_schema(
                available_tool.get("inputSchema", {})
            ),
            client=elitea,
            server=server_toolkit_name,
            tool_timeout_sec=toolkit_settings.get("timeout", 90)
        )
        # Inject display metadata so FE chips show human-readable names
        # and forward metadata into LangGraph run config via patched invoke().
        if tool.metadata is None:
            tool.metadata = {}
        if isinstance(tool.metadata, dict):
            tool.metadata['toolkit_type'] = 'mcp'
            tool.metadata['toolkit_name'] = toolkit_name or server_toolkit_name
            tool.metadata['display_name'] = toolkit_name or server_toolkit_name
        _patch_tool_invoke(tool)
        return tool
    except Exception as e:
        logger.error(f"Failed to create McpServerTool ('{server_toolkit_name}') for '{toolkit_name}.{tool_name}': {e}")
        return None

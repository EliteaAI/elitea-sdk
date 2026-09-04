import base64
import json
import logging
import textwrap
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from langchain_core.callbacks import dispatch_custom_event
from langchain_core.messages import ToolCall
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt

from ..exceptions import budget_exceeded_from
from ..tool_outcome import (
    ToolOutcome,
    ToolResultStatus,
    classify_tool_error,
    outcome_sink,
    record_outcome,
    retriable_for,
)
from ..utils.mcp_oauth import McpAuthorizationRequired
from typing import Any, Optional, Union
from langchain_core.utils.function_calling import convert_to_openai_tool

from ..langchain.utils import propagate_the_input_mapping, safe_serialize, object_to_dict, log_tool_result
from ..tool_result_bounds import bound_and_record, toolkit_type_of

logger = logging.getLogger(__name__)

# State key used to signal that a FunctionTool node was blocked
PIPELINE_BLOCKED_KEY = '_pipeline_blocked'

# State keys carrying the typed outcome of this node's tool call, so a pipeline can
# route on failure without pattern-matching the message history.
TOOL_OUTCOMES_KEY = 'tool_outcomes'
LAST_TOOL_OUTCOME_KEY = 'last_tool_outcome'

# Cap on the outcome message duplicated into state (routing needs status/error_class, never
# the full prose, which already lives on the message channel).
_MAX_OUTCOME_MESSAGE_CHARS = 4000

def replace_escaped_newlines(data):
    """
        Replace \\n with \n in all string values recursively.
        Required for sanitization of state variables in code node
    """
    if isinstance(data, dict):
        return {key: replace_escaped_newlines(value) for key, value in data.items()}
    elif isinstance(data, str):
        return data.replace('\\n', '\n')
    else:
        return data

class FunctionTool(BaseTool):
    name: str = 'FunctionalTool'
    description: str = 'This is direct call node for tools'
    tool: BaseTool = None
    return_type: str = "str"
    input_variables: Optional[list[str]] = None
    input_mapping: Optional[dict[str, dict]] = None
    output_variables: Optional[list[str]] = None
    structured_output: Optional[bool] = False
    elitea_client: Optional[Any] = None
    debug: bool = False
    auth_tool_name: Optional[str] = None
    auth_toolkit_name: Optional[str] = None
    auth_toolkit_type: Optional[str] = None

    def _prepare_pyodide_input(self, state: Union[str, dict, ToolCall], input_variables: Optional[list[str]] = None) -> str:
        """Prepare input for PyodideSandboxTool by injecting state into the code block.

        Logic for state variable injection:
        - If input_variables is None, empty, or only contains 'messages': inject ALL state variables
        - If input_variables contains specific variable names: inject ONLY those variables (excluding 'messages')

        Uses base64 encoding to avoid string escaping issues when passing JSON
        through multiple layers of parsing (Python -> Deno -> Pyodide) and compression to minimize args list
        """
        import base64
        import zlib

        state_copy = replace_escaped_newlines(deepcopy(state))

        # Always remove 'messages' from state injection
        if 'messages' in state_copy:
            del state_copy['messages']

        # Filter state variables based on input_variables
        # If no input_variables specified or only 'messages', include all state vars
        # Otherwise, include only specified variables (excluding 'messages')
        if input_variables is None or len(input_variables) == 0 or (len(input_variables) == 1 and input_variables[0] == 'messages'):
            # Include all state variables (messages already removed above)
            filtered_state = state_copy
            logger.debug("Code node: injecting ALL state variables into elitea_state")
        else:
            # Include only specified variables, excluding 'messages'
            filtered_state = {}
            for var in input_variables:
                if var != 'messages' and var in state_copy:
                    filtered_state[var] = state_copy[var]
            logger.debug(f"Code node: injecting ONLY specified variables into elitea_state: {list(filtered_state.keys())}")

        # Use safe_serialize to handle Pydantic models, datetime, and other non-JSON types
        filtered_state_dict = object_to_dict(filtered_state)
        state_json = safe_serialize(filtered_state_dict)

        # Use base64 encoding to avoid all string escaping issues
        # This is more robust than repr() when the code passes through multiple parsers
        # use compression to avoid issue with `{"error": "Error executing code: [Errno 7] Argument list too long: 'deno'"}`
        compressed = zlib.compress(state_json.encode('utf-8'))
        encoded = base64.b64encode(compressed).decode('ascii')

        pyodide_predata = f'''#state dict
import json
import base64
import zlib

compressed_state = base64.b64decode('{encoded}')
state_json = zlib.decompress(compressed_state).decode('utf-8')
elitea_state = json.loads(state_json)
# copies for backwards compatibility with old code that references alita_state and alita_client directly
alita_state = elitea_state.copy()
alita_client = elitea_client
'''
        return pyodide_predata

    def _handle_pyodide_output(self, tool_result: Any) -> dict:
        """Handle output processing for PyodideSandboxTool results."""
        tool_result_converted = {}

        if self.output_variables:
            for var in self.output_variables:
                if var == "messages":
                    tool_result_converted.update(
                        {"messages": [{"role": "assistant", "content": safe_serialize(tool_result)}]})
                    continue
                if isinstance(tool_result, dict) and var in tool_result:
                    tool_result_converted[var] = tool_result[var]
                else:
                    # handler in case user points to a var that is not in the output of the tool
                    tool_result_converted[var] = tool_result.get('result',
                                                                 tool_result.get('error') if tool_result.get('error')
                                                                 else 'Execution result is missing')
        else:
            tool_result_converted.update({"messages": [{"role": "assistant", "content": safe_serialize(tool_result)}]})

        if self.structured_output:
            # execute code tool and update state variables
            try:
                result_value = tool_result.get('result', {})
                if isinstance(result_value, dict):
                    tool_result_converted.update(result_value)
                elif isinstance(result_value, list):
                    # Handle list case - could wrap in a key or handle differently based on requirements
                    tool_result_converted.update({"result": result_value})
                else:
                    # Handle JSON string case
                    tool_result_converted.update(json.loads(result_value))
            except json.JSONDecodeError:
                logger.error(f"JSONDecodeError: {tool_result}")

        return tool_result_converted

    def _is_pyodide_tool(self) -> bool:
        """Check if the current tool is a PyodideSandboxTool."""
        return self.tool.name.lower() == 'pyodide_sandbox'

    def _build_client_preamble(self) -> str:
        """Build the sandbox client preamble that makes the saved file standalone-executable.

        Reads ``sandbox_client.py`` from ``runtime/clients/`` and prepends it with an
        ``elitea_client = SandboxClient(...)`` instantiation so the saved artifact can be
        run directly in a standard Python interpreter without the full SDK installed.

        Mirrors the logic in ``PyodideSandboxTool._prepare_pyodide_input()``.
        """
        try:
            sandbox_client_path = Path(__file__).parent.parent / 'clients' / 'sandbox_client.py'
            with open(sandbox_client_path, 'r', encoding='utf-8') as f:
                sandbox_client_code = f.read()
            client_init = (
                f"elitea_client = SandboxClient("
                f"base_url='{self.elitea_client.base_url}',"
                f"project_id={self.elitea_client.project_id},"
                f"auth_token='<YOUR_AUTH_TOKEN>')  # TODO: replace with your token before running\n"
            )
            return f"#elitea simplified client\n{sandbox_client_code}\n{client_init}\n"
        except FileNotFoundError:
            logger.warning(
                "[code-debug] sandbox_client.py not found — saved file will reference "
                "undefined elitea_client; add the SandboxClient definition manually."
            )
            return ""
        except Exception as exc:
            logger.warning("[code-debug] Could not build client preamble: %s", exc)
            return ""

    @staticmethod
    def _wrap_user_code_for_debug(code: str) -> str:
        """Wrap user code so top-level ``await`` runs under standard CPython.

        Pyodide compiles user code with ``PyCF_ALLOW_TOP_LEVEL_AWAIT``; CPython
        does not. To keep the saved debug file copy-paste-runnable via
        ``python file.py``, indent the user block into an ``async`` coroutine
        and drive it with ``asyncio.run()``. Sync code is unaffected.
        """
        indented = textwrap.indent(code, "    ")
        return (
            "import asyncio as _elitea_debug_asyncio\n"
            "\n"
            "async def _elitea_debug_main():\n"
            f"{indented}\n"
            "    return locals().get('result')\n"
            "\n"
            'if __name__ == "__main__":\n'
            "    _elitea_debug_result = _elitea_debug_asyncio.run(_elitea_debug_main())\n"
            "    print(_elitea_debug_result)\n"
        )

    def _save_code_to_artifact(self, code: str, node_name: str) -> None:
        """Save the fully-assembled, standalone-executable code to the 'code-debug' artifact bucket.

        The saved file is structured as:
          1. ``sandbox_client.py`` contents  (SandboxClient class definition)
          2. ``elitea_client = SandboxClient(...)``  (live credentials)
          3. State preamble  (elitea_state dict + ``alita_client = elitea_client``)
          4. User code

        This exactly mirrors what ``PyodideSandboxTool._prepare_pyodide_input()`` injects
        at execution time, so the saved file can be run as-is in a standard Python 3
        interpreter (``pip install requests chardet`` is the only external dependency).

        Filename is ``<node_name>__<YYYYMMDD_HHMMSS>.py`` (UTC timestamp), so every run
        creates a new snapshot instead of overwriting the previous one.
        """
        try:
            bucket = "code-debug"
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"{node_name}__{timestamp}.py"
            # Prepend the client preamble so the file is fully self-contained,
            # then wrap user code in an async runner so top-level await — valid
            # under Pyodide — also runs under standard CPython.
            full_code = (
                f"{self._build_client_preamble()}"
                f"{self._wrap_user_code_for_debug(code)}"
            )
            artifact = self.elitea_client.artifact(bucket)
            result = artifact.create(filename, full_code.encode('utf-8'))
            if 'error' in result:
                logger.warning(
                    "[code-debug] Failed to save code artifact for node '%s': %s",
                    node_name, result['error']
                )
            else:
                logger.debug(
                    "[code-debug] Saved code for node '%s' → %s/%s",
                    node_name, bucket, filename
                )
        except Exception as exc:
            logger.warning(
                "[code-debug] Could not save code artifact for node '%s': %s",
                node_name, exc
            )

    @staticmethod
    def _is_sensitive_tool_blocked(tool_result: Any) -> bool:
        """Check if the tool result is a sensitive-tool blocked payload."""
        if not isinstance(tool_result, str):
            return False
        try:
            parsed = json.loads(tool_result)
            return isinstance(parsed, dict) and parsed.get('type') == 'sensitive_tool_blocked'
        except (json.JSONDecodeError, TypeError, ValueError):
            return False

    def _build_blocked_termination(self, tool_result: str) -> dict:
        """Build a clean termination result when a FunctionTool's tool is blocked.

        Writes None/empty to all declared output variables so downstream nodes
        don't receive corrupt blocked-result JSON. Adds a clear assistant
        message and sets the ``_pipeline_blocked`` flag so the graph routes
        to END.
        """
        try:
            blocked = json.loads(tool_result)
        except (json.JSONDecodeError, TypeError):
            blocked = {}

        blocked_tool = blocked.get('blocked_tool_name', self.tool.name)
        toolkit_type = blocked.get('blocked_toolkit_type', '')
        node_name = self.name or ''

        # Build a clean, user-friendly markdown message
        parts = [f"**Pipeline stopped** — the action **{blocked_tool}**"]
        details = []
        if toolkit_type:
            details.append(f"toolkit type: *{toolkit_type}*")
        if node_name:
            details.append(f"node: *{node_name}*")
        if details:
            parts[0] += f" ({', '.join(details)})"
        parts[0] += " was **blocked** by user."

        parts.append(
            f"Downstream nodes that depend on `{blocked_tool}` output "
            f"were skipped to prevent invalid data."
        )
        parts.append(
            "> **Tip:** Regenerate this message to re-trigger the approval "
            "request and try again."
        )
        message = "\n\n".join(parts)

        result: dict[str, Any] = {
            "messages": [{"role": "assistant", "content": message}],
            PIPELINE_BLOCKED_KEY: message,
        }

        # Null-out all declared output variables
        if self.output_variables:
            for var in self.output_variables:
                if var != "messages":
                    result[var] = None

        logger.warning(
            "[PIPELINE] FunctionTool '%s' blocked — clean termination. Tool: %s",
            self.name, blocked_tool,
        )
        return result

    def _build_mcp_auth_skipped_termination(self, payload: dict) -> dict:
        """Stop a direct Pipeline Toolkit node cleanly after Skip."""
        tool_name = payload.get('tool_name') or self.auth_tool_name or self.tool.name
        toolkit_name = payload.get('toolkit_name') or self.auth_toolkit_name or 'Toolkit'
        message = (
            f"**Pipeline stopped** — authorization for **{toolkit_name}** "
            f"(tool: `{tool_name}`, node: *{self.name}*) was skipped.\n\n"
            f"Downstream nodes that depend on `{self.name}` output were not "
            "executed because the required toolkit is unavailable.\n\n"
            "> **Tip:** Authorize the toolkit and run the pipeline again."
        )
        result: dict[str, Any] = {
            "messages": [{"role": "assistant", "content": message}],
            PIPELINE_BLOCKED_KEY: message,
        }
        for var in self.output_variables or []:
            if var != 'messages':
                result[var] = None
        return result

    def _build_mcp_auth_refresh_termination(self, payload: dict) -> dict:
        """Fail closed if an authorized Toolkit was not loaded on rebuild."""
        tool_name = payload.get('tool_name') or self.auth_tool_name or self.tool.name
        toolkit_name = payload.get('toolkit_name') or self.auth_toolkit_name or 'Toolkit'
        message = (
            f"**Pipeline stopped** — authorization for **{toolkit_name}** "
            f"(tool: `{tool_name}`, node: *{self.name}*) completed, but the "
            "authorized Toolkit was not available in this pipeline instance.\n\n"
            "Downstream nodes were not executed with an unavailable tool.\n\n"
            "> **Tip:** Run the pipeline again to load the authorized Toolkit."
        )
        result: dict[str, Any] = {
            "messages": [{"role": "assistant", "content": message}],
            PIPELINE_BLOCKED_KEY: message,
        }
        for var in self.output_variables or []:
            if var != 'messages':
                result[var] = None
        return result

    def _build_mcp_auth_interrupt(
        self,
        exc: McpAuthorizationRequired,
        config: Optional[RunnableConfig],
    ) -> dict:
        """Create the durable guardrail payload for a direct Pipeline node."""
        metadata = exc.to_dict()
        tool_metadata = getattr(self.tool, 'metadata', None) or {}
        configurable = config.get('configurable', {}) if isinstance(config, dict) else {}
        return {
            'type': 'hitl',
            'interrupt_id': f'mcp_auth_{uuid4().hex}',
            'guardrail_type': 'mcp_auth',
            'node_name': self.name,
            'message': str(exc),
            'available_actions': ['authorize', 'skip'],
            'tool_name': (
                self.auth_tool_name
                or metadata.get('tool_name')
                or tool_metadata.get('tool_name')
                or self.tool.name
            ),
            'toolkit_name': (
                self.auth_toolkit_name
                or metadata.get('toolkit_name')
                or tool_metadata.get('toolkit_name')
                or ''
            ),
            'toolkit_type': (
                self.auth_toolkit_type
                or metadata.get('toolkit_type')
                or tool_metadata.get('toolkit_type')
                or ''
            ),
            # Direct graph nodes have no LLM tool-call id.  The node id is the
            # stable leaf invocation identity in this checkpointed pipeline.
            'tool_call_id': f'pipeline:{self.name}',
            'tool_args': {},
            'server_url': metadata.get('server_url'),
            'resource_metadata_url': metadata.get('resource_metadata_url'),
            'www_authenticate': metadata.get('www_authenticate'),
            'resource_metadata': metadata.get('resource_metadata'),
            'authorization_servers': metadata.get('authorization_servers'),
            'status': metadata.get('status'),
            'thread_id': configurable.get('thread_id'),
            'checkpoint_ns': configurable.get('checkpoint_ns') or '',
        }

    def invoke(
            self,
            state: Union[str, dict, ToolCall],
            config: Optional[RunnableConfig] = None,
            **kwargs: Any,
    ) -> Any:
        with outcome_sink() as sink:
            result = self._invoke_node(state, config, **kwargs)
        return self._with_outcome(result, sink)

    def _outcome_for(self, result: dict, sink: list) -> ToolOutcome:
        """The typed outcome of the call that produced ``result``."""
        tool_name = getattr(self.tool, 'name', None)
        toolkit_type = (getattr(self.tool, 'metadata', None) or {}).get('toolkit_type')
        if result.get(PIPELINE_BLOCKED_KEY):
            # A decline or an auth skip is a deliberate user choice, not a failure.
            return ToolOutcome(
                status=ToolResultStatus.BLOCKED,
                message=str(result[PIPELINE_BLOCKED_KEY]),
                tool_name=tool_name,
                toolkit_type=toolkit_type,
            )
        if sink:
            return sink[-1]
        return ToolOutcome(
            status=ToolResultStatus.SUCCESS,
            message='',
            tool_name=tool_name,
            toolkit_type=toolkit_type,
        )

    def _with_outcome(self, result: Any, sink: list) -> Any:
        """Add the outcome keys. Purely additive — declared output variables are untouched."""
        if not isinstance(result, dict):
            return result
        payload = self._outcome_for(result, sink).model_dump(mode='json')
        # message is LLM-facing prose already living on the message channel; routing only
        # ever needs status/error_class/retriable, so cap what gets duplicated into state.
        # Distinct keys from truncated/original_size: those describe the TOOL RESULT being
        # cut (status=TRUNCATED), a data-completeness signal for the pipeline author. This is
        # an unrelated storage detail about the envelope itself.
        message = payload.get('message') or ''
        if len(message) > _MAX_OUTCOME_MESSAGE_CHARS:
            payload['message_original_size'] = len(message)
            payload['message_truncated'] = True
            payload['message'] = message[:_MAX_OUTCOME_MESSAGE_CHARS]
        node_key = self.name or getattr(self.tool, 'name', '') or ''
        result[TOOL_OUTCOMES_KEY] = {node_key: payload}
        result[LAST_TOOL_OUTCOME_KEY] = payload
        return result

    def _invoke_node(
            self,
            state: Union[str, dict, ToolCall],
            config: Optional[RunnableConfig] = None,
            **kwargs: Any,
    ) -> Any:
        params = convert_to_openai_tool(self.tool).get(
            'function', {'parameters': {}}).get(
            'parameters', {'properties': {}}).get('properties', {})

        func_args = propagate_the_input_mapping(input_mapping=self.input_mapping, input_variables=self.input_variables,
                                                state=state)

        # For subgraph nodes (nested pipelines/agents), pass through state variables to the child.
        # Parent state values should override child's defaults when variable names collide.
        # This ensures that when parent assigns "foo = X" and child also has "foo", the child sees X.
        #
        # Check both is_subgraph attribute AND if the tool is an Application type
        # (Application tools represent nested agents/pipelines)
        is_nested_app = (
            (hasattr(self.tool, 'is_subgraph') and self.tool.is_subgraph) or
            type(self.tool).__name__ == 'Application'
        )
        if is_nested_app:
            logger.debug(f"[FUNC_TOOL] Passing state variables to nested app. State keys: {list(state.keys())}, func_args keys: {list(func_args.keys())}")
            for key, value in state.items():
                if key in ('messages', 'input', TOOL_OUTCOMES_KEY, LAST_TOOL_OUTCOME_KEY):
                    continue
                # Pass state variables to child, overriding input_mapping values
                # when the parent has explicitly set a value (not None/empty).
                # This fixes the issue where parent's assigned variable values
                # were not available in nested pipelines with same-named variables.
                if key not in func_args:
                    # Variable not in input_mapping - add it from parent state
                    logger.debug(f"[FUNC_TOOL] Adding '{key}' from state to func_args (value: {value!r})")
                    func_args[key] = value
                elif value is not None and value != '':
                    # Variable in input_mapping but parent has a real value - override
                    # This ensures parent's runtime value takes precedence over child's default
                    logger.debug(f"[FUNC_TOOL] Overriding '{key}' in func_args with state value (value: {value!r})")
                    func_args[key] = value
            logger.debug(f"[FUNC_TOOL] Final func_args keys: {list(func_args.keys())}")

        # special handler for PyodideSandboxTool
        if self._is_pyodide_tool():
            func_args['code'] = f"{self._prepare_pyodide_input(state, self.input_variables)}\n{func_args['code']}"
            # When debug mode is enabled and an elitea_client is available, persist
            # the full assembled code to the 'code-debug' artifact bucket so it can
            # be inspected after execution.
            if self.debug and self.elitea_client is not None:
                self._save_code_to_artifact(func_args['code'], self.name)

        try:
            tool_result = self.tool.invoke(func_args, config, **kwargs)
            tool_result = bound_and_record(
                tool_result, getattr(self.tool, 'name', None), toolkit_type_of(self.tool))

            # If the tool was blocked by the sensitive-tool guard (user
            # rejected), return a clean termination instead of propagating
            # the blocked-result JSON as a real tool output.
            if self._is_sensitive_tool_blocked(tool_result):
                return self._build_blocked_termination(tool_result)

            dispatch_custom_event(
                "on_function_tool_node", {
                    "input_mapping": self.input_mapping,
                    "input_variables": self.input_variables,
                    "state": state,
                    "tool_result": tool_result,
                }, config=config
            )
            _tool_meta = getattr(self.tool, 'metadata', None) or {}
            log_tool_result(logger, self.name, getattr(self.tool, 'name', None),
                            _tool_meta.get('toolkit_id'), tool_result)

            # handler for PyodideSandboxTool
            if self._is_pyodide_tool():
                return self._handle_pyodide_output(tool_result)

            # For nested Application tools (pipelines/agents), propagate ALL state variables
            # from child back to parent, not just those in output_variables.
            # This ensures bidirectional state flow between parent and child pipelines.
            if is_nested_app and isinstance(tool_result, dict):
                # Build result with standardized format - prefer 'output' key, fallback to 'messages'
                if 'output' in tool_result:
                    # Standardized format: use output directly, include messages if present
                    result_dict = {
                        "output": tool_result['output'],
                        "messages": tool_result.get('messages', [{"role": "assistant", "content": tool_result['output']}])
                    }
                elif 'messages' in tool_result:
                    # Legacy format: extract output from messages
                    result_dict = {"messages": tool_result['messages']}
                else:
                    result_dict = {
                        "messages": [{
                            "role": "assistant",
                            "content": safe_serialize(tool_result)
                        }]
                    }
                # Propagate all state variables from child (excluding internal keys).
                # For react agents (is_subgraph=False): also exclude declared output_variables
                # from propagation — they contain stale parent-state values that were injected
                # into the child at invocation time, and must be set from the agent's last AI
                # message instead.
                # For pipelines (is_subgraph=True): do NOT exclude output_variables — the child
                # pipeline may have explicitly computed and set them.
                is_pipeline = hasattr(self.tool, 'is_subgraph') and self.tool.is_subgraph
                excluded_keys = {'messages', 'output', 'input', 'chat_history',
                                  TOOL_OUTCOMES_KEY, LAST_TOOL_OUTCOME_KEY}
                if not is_pipeline:
                    # React agent: stale parent-state values for output vars must not leak back
                    excluded_keys.update(self.output_variables or [])
                for key, value in tool_result.items():
                    if key not in excluded_keys:
                        result_dict[key] = value
                        logger.debug(f"[FUNC_TOOL] Propagating '{key}' from nested app to parent state")

                # Map the agent/pipeline output to declared output_variables.
                if self.output_variables:
                    # Extract the agent's text content - prefer 'output' key, fallback to messages
                    agent_output_content = result_dict.get('output')
                    if agent_output_content is None:
                        messages = result_dict.get('messages', [])
                        if messages:
                            last_msg = messages[-1]
                            if isinstance(last_msg, dict):
                                agent_output_content = last_msg.get('content', '')
                            elif hasattr(last_msg, 'content'):
                                agent_output_content = last_msg.content

                    for var in self.output_variables:
                        if var == 'messages':
                            continue
                        if is_pipeline and var in result_dict and result_dict[var] is not None:
                            # Child pipeline explicitly set this variable — respect it
                            logger.debug(f"[FUNC_TOOL] Keeping child pipeline value for output_variable '{var}'")
                        else:
                            # React agent (or pipeline that didn't set the var) —
                            # use the agent's last AI message as the output
                            result_dict[var] = agent_output_content
                            logger.debug(
                                f"[FUNC_TOOL] Mapping agent text output to output_variable '{var}'"
                            )

                # The child ran in its own outcome sink, isolated from this node's — without
                # this, a child pipeline that ends on a failed tool call reports SUCCESS here.
                # Isolated in its own try: a malformed child payload must not be misread by
                # the outer `except Exception` as THIS node's own call having failed.
                child_outcome = tool_result.get(LAST_TOOL_OUTCOME_KEY)
                if isinstance(child_outcome, dict) and child_outcome.get('status') in (
                        ToolResultStatus.ERROR.value, ToolResultStatus.BLOCKED.value):
                    try:
                        record_outcome(ToolOutcome(**child_outcome))
                    except Exception:
                        logger.debug("[FUNC_TOOL] Malformed child last_tool_outcome, skipping relay", exc_info=True)

                return result_dict

            if not self.output_variables:
                return {"messages": [{"role": "assistant", "content": safe_serialize(tool_result)}]}
            else:
                if "messages" in self.output_variables:
                    if isinstance(tool_result, dict) and 'messages' in tool_result:
                        messages_dict = {"messages": tool_result['messages']}
                    else:
                        messages_dict = {
                            "messages": [{
                                "role": "assistant",
                                "content": safe_serialize(tool_result)
                                if not isinstance(tool_result, str)
                                else str(tool_result)
                            }]
                        }
                    for var in self.output_variables:
                        if var != "messages":
                            if isinstance(tool_result, dict) and var in tool_result:
                                messages_dict[var] = tool_result[var]
                            else:
                                messages_dict[var] = tool_result
                    return messages_dict
                else:
                    return { self.output_variables[0]: object_to_dict(tool_result) }
        except GraphBubbleUp:
            raise
        except McpAuthorizationRequired as auth_error:
            # Direct Toolkit/MCP pipeline nodes bypass the LLM tool loop, so
            # this checkpointed FunctionTool node owns their dynamic auth
            # guard.  A rebuilt continuation supplies the authorized real tool
            # at this same node; Skip terminates with the established polite
            # pipeline-stop contract.
            auth_payload = self._build_mcp_auth_interrupt(auth_error, config)
            resume_value = interrupt(auth_payload)
            action = (
                str((resume_value or {}).get('action') or 'skip').strip().lower()
                if isinstance(resume_value, dict)
                else 'skip'
            )
            if action != 'authorize':
                return self._build_mcp_auth_skipped_termination(auth_payload)
            # This branch is a defensive same-runnable fallback.  Production
            # continuations rebuild the application with the new OAuth token,
            # so the real tool executes before reaching this catch.
            return self._build_mcp_auth_refresh_termination(auth_payload)
        # save the whole error message to the tool's output.
        # Behavior change (#6171): an unwrapped ValueError used to re-raise as a bare
        # ToolException here; it now resolves gracefully like every other local failure.
        except Exception as e:
            # A budget rejection is not a tool-input problem and cannot be retried
            budget_error = budget_exceeded_from(e)
            if budget_error is not None:
                raise budget_error from e
            # Reached only when the middleware never wrapped this tool, so classify here.
            error_class = classify_tool_error(e)
            record_outcome(ToolOutcome(
                status=ToolResultStatus.ERROR,
                message=str(e),
                tool_name=getattr(self.tool, 'name', None),
                error_class=error_class,
                retriable=retriable_for(error_class),
                exception_type=type(e).__name__,
                retry_after=getattr(e, 'retry_after', None),
            ))
            return {"messages": [
                {"role": "assistant", "content": f"""Tool input to the {self.tool.name} with value {func_args} raised Exception.
                        \n\nTool schema is {safe_serialize(params)}. \n\n Details: {e}"""}]}

    def _run(self, *args, **kwargs):
        return self.invoke(**kwargs)

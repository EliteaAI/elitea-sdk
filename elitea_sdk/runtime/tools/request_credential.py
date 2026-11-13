"""Request-Credential tool (C-1009, EliteaAI/elitea_issues#7228).

A governed agent that needs a credential asks the person for it through the chat's credential form instead of
handling the value itself: the tool takes only the *shape* of the request (the credential type, a title, an optional
purpose) and pauses the run with a ``credential_form`` interrupt. The person types the value into the form in the
chat; Elitea Main stores it as a governed credential and resumes the run with the server's own record of it (reference
id, title, type, state), never the value. This tool turns that record into a fixed sentence for the model; any other
key is dropped.

The tool is given to the model only when Main stamped ``credential_form: true`` on the run's internal Elitea MCP
toolkit (a governed run whose token scope holds ``credentials``) and the run token applies to that toolkit; never under
governance off. It reuses the HITL pause/resume of ``ask_user`` (``interrupt()`` / ``Command(resume=...)``).
"""

import logging
from typing import Any, List, Optional, Type
from uuid import uuid4

from langchain_core.messages.base import message_to_dict
from langchain_core.tools import BaseTool
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

CREDENTIAL_FORM_GUARDRAIL = "credential_form"
REQUEST_CREDENTIAL_TOOL_NAME = "request_credential"
CREDENTIAL_FORM_SETTING = "credential_form"
CREATED_ACTION = "created"
CANCEL_ACTION = "cancel"
# What the server's record of a created credential may carry; anything else is dropped (and logged by name only)
RESULT_KEYS = ("reference_id", "title", "credential_type", "state", "configuration_uuid", "configuration_title")
MAX_TEXT = 200

REQUEST_CREDENTIAL_DESCRIPTION = (
    "Ask the user to create a credential (an API token, a password, a Microsoft sign-in) you need for a toolkit. "
    "Never ask the user to paste a secret into the chat and never put a secret into a tool call yourself: this tool "
    "shows the user a secure form in the chat, the user types the value there, and you get back only the new "
    "credential's reference, title, type and state. Give the credential type exactly as the platform names it "
    "(for example 'github', 'jira', 'sharepoint'), a short title and, optionally, what it is for."
)


class RequestCredentialInput(BaseModel):
    """The shape of the request, nothing more: no field can carry a value."""
    model_config = ConfigDict(extra="forbid")

    credential_type: str = Field(min_length=1, max_length=64,
                                 description="The credential type as the platform names it, e.g. 'github'.")
    title: str = Field(min_length=1, max_length=64, description="A short title for the new credential.")
    purpose: Optional[str] = Field(default=None, max_length=MAX_TEXT,
                                   description="Optional: what the credential is needed for (no secret).")


def _clean(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def format_result(resume_value: Any) -> str:
    """The tool result the model reads for a resume: one fixed sentence built from the allowed keys only."""
    value = resume_value if isinstance(resume_value, dict) else {}
    action = value.get("action")
    if action != CREATED_ACTION:
        return "The user did not provide the credential."
    record = value.get("value") if isinstance(value.get("value"), dict) else {}
    dropped = sorted(key for key in record if key not in RESULT_KEYS)
    if dropped:
        logger.warning("[REQUEST_CREDENTIAL] dropped unexpected resume keys: %s", ", ".join(map(str, dropped)))
    title = _clean(record.get("title"), 64)
    kind = _clean(record.get("credential_type"), 64)
    reference = _clean(record.get("reference_id"), 64)
    state = _clean(record.get("state"), 32)
    if not reference:
        return "The user did not provide the credential."
    text = (f"Credential '{title}' ({kind}) is registered as reference {reference}, state {state}; a grant for you "
            f"is requested and an administrator must approve it before the agent can use it.")
    configuration_uuid = _clean(record.get("configuration_uuid"), 64)
    configuration_title = _clean(record.get("configuration_title"), 128)
    if configuration_uuid and configuration_title:
        text += (f" A toolkit selects it as the configuration '{configuration_title}' (uuid {configuration_uuid}), "
                 "never by its value.")
    return text


class RequestCredentialTool(BaseTool):
    """Pause the agent and ask the user to create a credential through the chat's credential form."""

    name: str = REQUEST_CREDENTIAL_TOOL_NAME
    description: str = REQUEST_CREDENTIAL_DESCRIPTION
    args_schema: Type[BaseModel] = RequestCredentialInput

    def _capture_pending_messages(self) -> List[dict]:
        """As ``ask_user``: completed sibling tool messages survive the pause (they are not re-run on resume)."""
        try:
            from .llm import _PENDING_TOOL_MESSAGES
        except Exception:  # pragma: no cover - defensive
            return []
        serialized: List[dict] = []
        for msg in _PENDING_TOOL_MESSAGES.get([]) or []:
            try:
                serialized.append(message_to_dict(msg))
            except Exception:
                pass
        return serialized

    def _run(self, credential_type: str = "", title: str = "", purpose: Optional[str] = None,
             run_manager: Any = None, **kwargs: Any) -> str:
        request = RequestCredentialInput(credential_type=credential_type, title=title, purpose=purpose)
        args = request.model_dump()
        interrupt_payload = {
            "type": "hitl",
            "interrupt_id": f"hitl_{uuid4().hex}",
            "guardrail_type": CREDENTIAL_FORM_GUARDRAIL,
            "node_name": REQUEST_CREDENTIAL_TOOL_NAME,
            "tool_name": REQUEST_CREDENTIAL_TOOL_NAME,
            "credential_type": request.credential_type,
            "title": request.title,
            "purpose": request.purpose,
            "message": f"The agent asks you to create the credential '{request.title}' ({request.credential_type}).",
            "available_actions": [CREATED_ACTION, CANCEL_ACTION],
            "routes": {},
            # echoed for the graph resume handler to re-issue this exact tool call (the three shape fields only)
            "tool_args": args,
        }
        pending = self._capture_pending_messages()
        if pending:
            interrupt_payload["_pending_messages"] = pending
        logger.info("[REQUEST_CREDENTIAL] Interrupting for a %s credential form.", request.credential_type)
        resume_value = interrupt(interrupt_payload)
        result = format_result(resume_value)
        logger.info("[REQUEST_CREDENTIAL] Resumed (%s).",
                    resume_value.get("action") if isinstance(resume_value, dict) else "no action")
        return result

    async def _arun(self, credential_type: str = "", title: str = "", purpose: Optional[str] = None,
                    run_manager: Any = None, **kwargs: Any) -> str:
        return self._run(credential_type=credential_type, title=title, purpose=purpose, run_manager=run_manager)


def credential_form_requested(tools_list: Any, client: Any) -> bool:
    """Did Main stamp ``credential_form: true`` on an internal Elitea MCP toolkit the run token of this run applies to?
    Without a run token (governance off, a legacy run) never."""
    from ..utils.internal_run_auth import internal_run_headers
    for tool in tools_list or []:
        if not isinstance(tool, dict) or not str(tool.get("type", "")).startswith("mcp"):
            continue
        settings = tool.get("settings") if isinstance(tool.get("settings"), dict) else {}
        if settings.get(CREDENTIAL_FORM_SETTING) is not True or not isinstance(settings.get("url"), str):
            continue
        _, identity = internal_run_headers(client, settings["url"], {})
        if identity is not None:
            return True
    return False

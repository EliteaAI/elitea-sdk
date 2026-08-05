"""Ask-User (clarifying question) tool.

An LLM-callable tool that pauses the agent mid-run to ask the user a structured
clarifying question (multiple choice + optional free-text "Other"), then resumes
with the user's answer injected back as the tool result.

It reuses the platform's existing HITL pause/resume pipeline
(LangGraph ``interrupt()`` / ``Command(resume=...)``, checkpointed by
``thread_id``). Because ``interrupt()`` fires from inside the single LLM+tools
graph node, the graph-level resume handler recognizes
``guardrail_type == 'clarifying_question'`` and rebuilds a synthetic AIMessage
that re-issues this exact tool call on resume (mirroring the sensitive-tool
guard) so the LLM is never re-invoked non-deterministically and this tool's
``interrupt()`` receives the answer.
"""

import json
import logging
from typing import Any, List, Optional, Type
from uuid import uuid4

from langchain_core.messages.base import message_to_dict
from langchain_core.tools import BaseTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CLARIFYING_QUESTION_GUARDRAIL = "clarifying_question"
ASK_USER_TOOL_NAME = "ask_user"
ASK_USER_ANSWER_ACTION = "answer"

ASK_USER_DESCRIPTION = (
    "Ask the user a clarifying question when you lack information or are "
    "uncertain how to proceed, instead of guessing. Presents 1-4 questions, "
    "each with a short header and a list of selectable options; the user picks "
    "an option (or types their own answer when allowed) and execution resumes "
    "with their answer. Use this for genuine decision points — which approach, "
    "which target, ambiguous requirements — not for information you can find "
    "yourself. Do NOT use it to ask permission to run a tool."
)


class AskUserOption(BaseModel):
    label: str = Field(description="Concise choice label shown to the user.")
    description: str = Field(
        default="",
        description="Optional short explanation of what choosing this option means.",
    )


class AskUserQuestionSpec(BaseModel):
    question: str = Field(description="The question to ask the user.")
    header: str = Field(
        default="",
        description="Very short label for the question (a few words).",
    )
    options: List[AskUserOption] = Field(
        default_factory=list,
        description="Selectable options. Provide 2-4 distinct choices when possible.",
    )
    multi_select: bool = Field(
        default=False,
        description="Allow the user to select more than one option.",
    )
    allow_other: bool = Field(
        default=True,
        description="Render a free-text 'Other' input so the user can type their own answer.",
    )


class AskUserInput(BaseModel):
    questions: List[AskUserQuestionSpec] = Field(
        description="1-4 questions to ask the user at once.",
    )


def _normalize_questions(questions: Any) -> List[dict]:
    """Coerce validated/raw question specs into JSON-serializable payload dicts."""
    normalized: List[dict] = []
    for idx, q in enumerate(questions or []):
        if isinstance(q, AskUserQuestionSpec):
            q = q.model_dump()
        elif not isinstance(q, dict):
            continue
        options = []
        for opt in q.get("options", []) or []:
            if isinstance(opt, AskUserOption):
                opt = opt.model_dump()
            if isinstance(opt, dict) and opt.get("label"):
                options.append({
                    "label": str(opt["label"]),
                    "description": str(opt.get("description", "") or ""),
                })
            elif isinstance(opt, str) and opt:
                options.append({"label": opt, "description": ""})
        normalized.append({
            "id": str(q.get("id") or f"q{idx + 1}"),
            "question": str(q.get("question", "") or ""),
            "header": str(q.get("header", "") or ""),
            "options": options,
            "multiSelect": bool(q.get("multi_select", q.get("multiSelect", False))),
            "allow_other": bool(q.get("allow_other", True)),
        })
    return normalized


def _format_answer(questions: List[dict], resume_value: Any) -> str:
    """Format the user's resume answer into a concise tool result for the LLM."""
    if isinstance(resume_value, dict):
        answers = resume_value.get("value", resume_value)
    else:
        answers = resume_value
    if not isinstance(answers, dict):
        text = str(answers or "").strip()
        return f"User answered: {text}" if text else "User did not provide an answer."

    lines: List[str] = []
    by_id = {q["id"]: q for q in questions}
    for key, val in answers.items():
        label = by_id.get(key, {}).get("question") or by_id.get(key, {}).get("header") or key
        if isinstance(val, (list, tuple)):
            rendered = ", ".join(str(v) for v in val if str(v).strip())
        else:
            rendered = str(val)
        if rendered.strip():
            lines.append(f"- {label}: {rendered.strip()}")
    if not lines:
        return "User did not provide an answer."
    return "User answered:\n" + "\n".join(lines)


class AskUserTool(BaseTool):
    """Pause the agent and ask the user a structured clarifying question."""

    name: str = ASK_USER_TOOL_NAME
    description: str = ASK_USER_DESCRIPTION
    args_schema: Type[BaseModel] = AskUserInput
    # When True (non-interactive / API-only runs) the tool never pauses; it
    # returns immediately so headless executions cannot hang on a human.
    auto_skip: bool = False

    def _capture_pending_messages(self) -> List[dict]:
        """Serialize in-flight tool messages so completed work survives the pause.

        Mirrors the sensitive-tool guard: __perform_tool_calling stores the
        accumulated intermediate messages in a contextvar before invoking each
        tool; embedding them in the interrupt payload lets the graph-level resume
        handler restore them so already-completed sibling tools are not re-run.
        """
        try:
            from .llm import _PENDING_TOOL_MESSAGES
        except Exception:  # pragma: no cover - defensive
            return []
        pending = _PENDING_TOOL_MESSAGES.get([])
        serialized: List[dict] = []
        for msg in pending or []:
            try:
                serialized.append(message_to_dict(msg))
            except Exception:
                pass
        return serialized

    def _run(self, questions: Any = None, run_manager: Any = None, **kwargs: Any) -> str:
        normalized = _normalize_questions(questions if questions is not None else kwargs.get("questions"))
        if not normalized:
            return (
                "No questions were provided to ask_user. Proceed using your best "
                "judgement or state what information you still need."
            )

        if self.auto_skip:
            logger.info("[ASK_USER] auto_skip enabled (non-interactive run) — not pausing.")
            return (
                "The user is not available to answer in this non-interactive run. "
                "Proceed with a reasonable default and state the assumption you made."
            )

        message = normalized[0].get("question") or "Please answer to continue."
        interrupt_payload = {
            "type": "hitl",
            "interrupt_id": f"hitl_{uuid4().hex}",
            "guardrail_type": CLARIFYING_QUESTION_GUARDRAIL,
            "node_name": ASK_USER_TOOL_NAME,
            "tool_name": ASK_USER_TOOL_NAME,
            "message": message,
            "questions": normalized,
            "available_actions": [ASK_USER_ANSWER_ACTION],
            "routes": {},
            # Echoed for the graph resume handler to rebuild the synthetic
            # AIMessage that re-issues this exact tool call on resume.
            "tool_args": {"questions": normalized},
        }

        pending = self._capture_pending_messages()
        if pending:
            interrupt_payload["_pending_messages"] = pending

        logger.info("[ASK_USER] Interrupting with %d clarifying question(s).", len(normalized))
        resume_value = interrupt(interrupt_payload)
        logger.info("[ASK_USER] Resumed with: %s", resume_value)
        return _format_answer(normalized, resume_value)

    async def _arun(self, questions: Any = None, run_manager: Any = None, **kwargs: Any) -> str:
        return self._run(questions=questions, run_manager=run_manager, **kwargs)

"""On-demand Project Context loading for agent and direct-chat runtimes."""

import logging
from hashlib import sha256
from typing import List, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .base import Middleware


logger = logging.getLogger(__name__)


class ReadProjectContextInput(BaseModel):
    """The Project Context loader intentionally takes no arguments."""


class ReadProjectContextTool(BaseTool):
    name: str = "read_project_context"
    description: str
    args_schema: type[BaseModel] = ReadProjectContextInput
    context_content: str = Field(exclude=True)
    revision: str = Field(exclude=True)
    metadata: dict = Field(default_factory=lambda: {
        "toolkit_type": "internal",
        "toolkit_name": "project_context",
        "display_name": "Project Context",
    })

    def _read(self) -> str:
        logger.info(
            "Project Context loaded on demand: revision=%s context_chars=%d",
            self.revision,
            len(self.context_content),
        )
        return (
            f"Project Context revision: {self.revision}\n\n"
            f"# Project Context\n\n{self.context_content}"
        )

    def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self._read()

    async def _arun(self, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self._read()


class ProjectContextMiddleware(Middleware):
    """Expose one always-bound tool whose schema carries the activation intent."""

    def __init__(self, project_context: dict, conversation_id: Optional[str] = None):
        super().__init__(conversation_id=conversation_id)
        content = str((project_context or {}).get('content') or '').strip()
        activation_description = ' '.join(
            str((project_context or {}).get('activation_description') or '').split()
        )
        if not content or not activation_description:
            raise ValueError("Project Context content and activation_description are required")

        self.content = content
        self.activation_description = activation_description[:300]
        self.revision = str(
            (project_context or {}).get('revision')
            or sha256(content.encode('utf-8')).hexdigest()
        )
        self._tool: Optional[ReadProjectContextTool] = None
        logger.info(
            "Project Context available on demand: revision=%s context_chars=%d activation_chars=%d",
            self.revision,
            len(self.content),
            len(self.activation_description),
        )

    def get_tools(self) -> List[BaseTool]:
        if self._tool is None:
            self._tool = ReadProjectContextTool(
                context_content=self.content,
                revision=self.revision,
                description=(
                    "Load the current Project Context before answering when the user's request "
                    f"matches this activation description: {self.activation_description} "
                    f"The current revision is {self.revision}. If conversation history contains "
                    "a different revision, call this tool again. Skip it when the request does "
                    "not match. Treat the returned content as project-specific background and "
                    "constraints, subject to higher-priority instructions."
                ),
            )
        return [self._tool]

    def get_system_prompt(self) -> str:
        return (
            "Before answering, check whether the user's current request matches the "
            "activation description of the `read_project_context` tool. You MUST call "
            "that tool before answering every matching request, and skip it for unrelated "
            "requests. Never apply Project Context unless you have loaded it."
        )

    @staticmethod
    def _is_project_context_result(message) -> bool:
        if not isinstance(message, ToolMessage):
            return False
        return (
            getattr(message, "name", None) == "read_project_context"
            or str(message.content).startswith("Project Context revision:")
        )

    def transform_messages_for_model(self, messages: list, config: dict) -> Optional[list]:
        """Hide prior full context results without changing checkpoint structure."""
        if not messages:
            return None

        keep_latest = self._is_project_context_result(messages[-1])
        transformed = []
        changed = False
        for index, message in enumerate(messages):
            is_latest = keep_latest and index == len(messages) - 1
            if self._is_project_context_result(message) and not is_latest:
                previous_revision = str(message.content).splitlines()[0].removeprefix(
                    "Project Context revision: "
                )
                transformed.append(message.model_copy(update={
                    "content": (
                        f"Previous Project Context revision: {previous_revision}. "
                        "Its full content is not loaded for this turn. "
                        f"The current revision is {self.revision}; call read_project_context "
                        "if the current request matches its activation description."
                    ),
                }))
                changed = True
            else:
                transformed.append(message)
        return transformed if changed else None

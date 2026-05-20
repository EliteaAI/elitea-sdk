"""
Standardized response format for SDK agent types.

All agent types (LangGraphAgentRunnable, SwarmResultAdapter, Application)
should return an AgentResponse to ensure consistent behavior across the platform.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """
    Standardized response format for all SDK agent types.

    This contract ensures consistent response handling across:
    - LangGraphAgentRunnable (react agents, pipelines)
    - SwarmResultAdapter (swarm mode)
    - Application._run() (nested agent/pipeline invocations)

    The pylon's extract_response_content() relies on 'output' being present.
    """

    output: str = Field(
        default="",
        description="Final text response. Always present, always a string."
    )
    messages: list = Field(
        default_factory=list,
        description="Message history. Always present, may be empty list."
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="Thread ID for conversation continuation. None if execution finished."
    )
    execution_finished: bool = Field(
        default=True,
        description="Whether the graph reached END node."
    )

    # Optional metadata fields
    context_info: Optional[dict] = Field(
        default=None,
        description="Summarization metadata (token counts, message counts)."
    )
    hitl_interrupt: Optional[dict] = Field(
        default=None,
        description="Human-in-the-loop interrupt metadata."
    )

    model_config = {
        "extra": "allow",  # Allow additional state variables to pass through
    }

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dict, including any extra fields.

        This ensures all state variables (both defined and extra) are included.
        """
        return self.model_dump(exclude_none=False)

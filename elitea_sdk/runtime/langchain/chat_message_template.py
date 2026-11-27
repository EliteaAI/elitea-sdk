import logging
from typing import Any, Dict, List
from jinja2 import DebugUndefined
from jinja2.exceptions import SecurityError
from ..utils.safe_templates import HardenedSandboxedEnvironment
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts.chat import BaseMessagePromptTemplate, BaseChatPromptTemplate

logger = logging.getLogger(__name__)

class Jinja2TemplatedChatMessagesTemplate(ChatPromptTemplate):

    def _resolve_variables(self, message: BaseMessage, kwargs: Dict) -> BaseMessage:
        # Sandboxed (#7681); a template the sandbox rejects keeps its raw content.
        environment = HardenedSandboxedEnvironment(undefined=DebugUndefined)
        try:
            content = environment.from_string(message.content).render(kwargs)
        except SecurityError:
            logger.warning("Message template rejected by sandbox; using raw content")
            content = message.content
        if isinstance(message, SystemMessage):
            return SystemMessage(content=content)
        elif isinstance(message, AIMessage):
            return AIMessage(content=content)
        elif isinstance(message, HumanMessage):
            return HumanMessage(content=content)
        elif isinstance(message, ToolMessage):
            return ToolMessage(content=content)
        else:
            return BaseMessage(content=content)

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format the chat template into a list of finalized messages.

        Args:
            **kwargs: keyword arguments to use for filling in template variables
                      in all the template messages in this chat template.

        Returns:
            list of formatted messages
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        result = []
        for message_template in self.messages:
            if isinstance(message_template, BaseMessage):
                message = self._resolve_variables(message_template, kwargs)
                logger.debug(message.content)
                result.append(message)
            elif isinstance(message_template, 
                            (BaseMessagePromptTemplate, BaseChatPromptTemplate)
                            ):
                message = message_template.format_messages(**kwargs)
                result.extend(message)
        return result
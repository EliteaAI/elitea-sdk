import logging
from json import dumps
from traceback import format_exc
from typing import Any, Optional, Union

from langchain_core.callbacks import dispatch_custom_event
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, ToolCall
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError, BaseModel, create_model

from .application import Application
from ..langchain.utils import _extract_json, log_tool_result
from ..tool_result_bounds import bound_and_record, toolkit_type_of
from ..tool_outcome import run_failure_message
from ...tools.utils.serialization import serialize_tool_result
from ..utils.safe_templates import safe_format

logger = logging.getLogger(__name__)


def _schema_rejections(error: Exception, schema_name: Optional[str]) -> dict:
    """What the LLM sent for each failing field when its structured output does not fit the tool's schema."""
    validation = error if isinstance(error, ValidationError) else error.__cause__
    if not isinstance(validation, ValidationError) or validation.title != schema_name:
        return {}
    return {'.'.join(map(str, item['loc'])): 'missing' if item['type'] == 'missing' else item['input']
            for item in validation.errors()}


class ToolNode(BaseTool):
    name: str = 'ToolNode'
    description: str = 'This is tool node for tools'
    client: Any = None
    tool: BaseTool = None
    return_type: str = "str"
    input_variables: Optional[list[str]] = None
    output_variables: Optional[list[str]] = None
    structured_output: Optional[bool] = False
    task: Optional[str] = None
    prompt: str = """You are tasked to formulate arguments for the tool according to user task and conversation history.
Tool name: {tool_name}
Tool description: {tool_description}
Tool arguments schema: 
{schema}

{task}  

"""
    unstructured_output: str = """Expected output is JSON that to be used as a KWARGS for the tool call like {{"key": "value"}} 
in case your key is "chat_history" value should be a list of messages with roles like {{"chat_history": [{{"role": "user", "content": "input"}}, {{"role": "assistant", "content": "output"}}]}}.
Tool won't have access to conversation so all keys and values need to be actual and independent.
Answer must be JSON only extractable by JSON.LOADS."""

    def _structured_output_schema(self):
        # Every field is required (defaults cut) and chat_history is dropped, for
        # application as a tool. A dict args_schema (MCP tools) comes back as a dict.
        args_schema = self.tool.args_schema
        if isinstance(args_schema, dict):
            properties = {name: prop for name, prop in (args_schema.get("properties") or {}).items()
                          if name != 'chat_history'}
            return {**args_schema, "title": "NewModel", "properties": properties, "required": list(properties)}
        fields = {name: (field.annotation, ...) for name, field
                  in args_schema.model_fields.items() if name != 'chat_history'}
        return create_model('NewModel', **fields)

    def invoke(
            self,
            state: Union[str, dict, ToolCall],
            config: Optional[RunnableConfig] = None,
            **kwargs: Any,
    ) -> Any:
        params = convert_to_openai_tool(self.tool).get(
            'function', {'parameters': {}}).get(
            'parameters', {'properties': {}}).get('properties', {})
        # this is becasue messages is shared between all tools and we need to make sure that we are not modifying it
        input_ = []
        last_message = {}
        logger.debug(f"ToolNode input: {self.input_variables}")
        logger.debug(f"Output variables: {self.output_variables}")
        for var in self.input_variables:
            if 'messages' in self.input_variables:
                messages = state.get('messages', [])[:]
                input_ = messages[:-1]
                last_message["user_input"] = messages[-1].content
            else:
                last_message[var] = state[var]
        logger.debug(f"ToolNode input: {input_}")
        if self.task:
            task = safe_format(self.task, **last_message, last_message=dumps(last_message))
        else:
            task = 'Input from user: {last_message}'.format(last_message=dumps(last_message))
        input_ += [
            HumanMessage(self.prompt.format(
                tool_name=self.tool.name,
                tool_description=self.tool.description,
                schema=params,
                task=task
            ))
        ]
        if self.structured_output:
            output_schema = self._structured_output_schema()
            llm = self.client.with_structured_output(output_schema)
            try:
                completion = llm.invoke(input_, config=config)
            except (ValidationError, OutputParserException) as error:
                rejected = _schema_rejections(error, getattr(output_schema, '__name__', None))
                if not rejected:
                    raise
                return self._invalid_input_reply(rejected, params, input_)
            result = completion if isinstance(completion, dict) else completion.model_dump()
        else:
            input_[-1].content += self.unstructured_output
            completion = self.client.invoke(input_, config=config)
            from ..langchain.utils import extract_text_from_completion
            content_text = extract_text_from_completion(completion)
            result = _extract_json(content_text.strip())
            _tk = (getattr(self.tool, 'metadata', None) or {}).get('toolkit_id')
            log_tool_result(logger, self.name, getattr(self.tool, 'name', None),
                            _tk, result, label='tool params')
        try:
            tool_result = self.tool.invoke(result, config=config, kwargs=kwargs)
            # Before the event dispatch: the event payload is what reaches the DB.
            tool_result = bound_and_record(
                tool_result, getattr(self.tool, 'name', None), toolkit_type_of(self.tool))
            dispatch_custom_event(
                "on_tool_node", {
                    "input_variables": self.input_variables,
                    "tool_result": tool_result,
                    "state": state,
                }, config=config
            )
            message_result = serialize_tool_result(tool_result)
            _tk = (getattr(self.tool, 'metadata', None) or {}).get('toolkit_id')
            log_tool_result(logger, self.name, getattr(self.tool, 'name', None),
                            _tk, tool_result)
            if not self.output_variables:
                return {"messages": [{"role": "assistant", "content": message_result}]}
            else:
                return {self.output_variables[0]: tool_result,
                        "messages": [{"role": "assistant", "content": message_result}]}
        except ValidationError:
            return self._invalid_input_reply(result, params, input_)

    def _invalid_input_reply(self, value: Any, params: dict, input_: list) -> dict:
        """Graceful node reply when the tool input fails validation."""
        logger.error(f"ValidationError: {format_exc()}")
        return {
            "messages": [run_failure_message(f"""Tool input to the {self.tool.name} with value {value} raised ValidationError. 
        \n\nTool schema is {dumps(params)} \n\nand the input to LLM was 
        {input_[-1].content}""")]}

    def _run(self, *args, **kwargs):
        return self.invoke(**kwargs)

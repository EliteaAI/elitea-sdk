from traceback import format_exc
from typing import List, Dict
from abc import ABCMeta
from jinja2 import TemplateSyntaxError, UndefinedError
from jinja2.exceptions import SecurityError
from .safe_templates import HardenedSandboxedEnvironment, TemplateRejectedError
import logging
from langgraph.graph import END

logger = logging.getLogger(__name__)

class TransformationError(Exception):
    "Raised when transformation fails"


class MyABC(ABCMeta):
    meta_registry = {}

    def __new__(mcs, name, bases, attrs):
        resulting_class = super().__new__(mcs, name, bases, attrs)
        if bases:  # exlcuding parent class
            name = name.split('TransformerEvaluate')[0].lower()
            mcs.meta_registry[name] = resulting_class
            resulting_class._output_format = name
        return resulting_class


class EvaluateTemplate(metaclass=MyABC):
    def __init__(self, query: str, context: Dict):
        self.query = query
        self.context = context

    def extract(self):
        # Sandboxed: condition text is pipeline-author input (#7681).
        environment = HardenedSandboxedEnvironment()

        def json_loads_filter(json_string: str,do_replace: bool = False):
            import json
            if do_replace:
                json_string = json_string.replace("'", "\"")
            return json.loads(json_string)

        environment.filters['json_loads'] = json_loads_filter
        try:
            template = environment.from_string(self.query)
            logger.info(f"Condition context: {self.context}")
            result = template.render(**self.context)
        except TemplateSyntaxError as exc:
            logger.critical(format_exc())
            logger.info('Template str: %s', self.query)
            raise Exception(f"Invalid jinja template: {exc.message}")
        except SecurityError:
            logger.warning('Template rejected by sandbox: %.200s', self.query)
            raise TemplateRejectedError("Template rejected: unsafe attribute access is not allowed")
        except UndefinedError as exc:
            # The template is valid; the data it referenced is missing. Saying "invalid
            # template" here sends the author to debug the wrong thing (issue #6171).
            logger.critical(format_exc())
            logger.info('Template str: %s', self.query)
            raise Exception(
                f"Jinja template referenced a value that is not available: {exc}. "
                f"Available names: {sorted(self.context)}"
            )
        return result
        
    # template method
    def evaluate(self):
        # extracting
        value: List[Dict] = self.extract()
        if 'END' in value.strip():
            return END
        else:
            return value.strip()
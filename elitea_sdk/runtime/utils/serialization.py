"""
Serialization utilities for safe JSON encoding of complex objects.

Handles Pydantic models, LangChain messages, datetime objects, and other
non-standard types that may appear in state variables.
"""
import json
import logging
from datetime import datetime, date
from typing import Any

logger = logging.getLogger(__name__)

# Import PydanticUndefined for handling unset field defaults
# This is needed to prevent serialization errors in LangGraph checkpoints
try:
    from pydantic_core import PydanticUndefined
    _PYDANTIC_UNDEFINED = PydanticUndefined
except ImportError:
    _PYDANTIC_UNDEFINED = None


def _is_pydantic_undefined(obj: Any) -> bool:
    """
    Check if an object is PydanticUndefined (Pydantic's sentinel for unset defaults).

    This handles the case where PydanticUndefined ends up in state and causes
    serialization errors in LangGraph checkpoint system (msgpack).

    Args:
        obj: Any object to check

    Returns:
        True if the object is PydanticUndefined, False otherwise
    """
    if _PYDANTIC_UNDEFINED is not None and obj is _PYDANTIC_UNDEFINED:
        return True
    # Fallback check by type name for edge cases
    return type(obj).__name__ == 'PydanticUndefinedType'


def _convert_to_serializable(obj: Any, _seen: set = None) -> Any:
    """
    Recursively convert an object to JSON-serializable primitives.

    Handles nested dicts and lists that may contain non-serializable objects.
    Uses a seen set to prevent infinite recursion with circular references.

    Args:
        obj: Any object to convert
        _seen: Internal set to track seen object ids (for circular reference detection)

    Returns:
        JSON-serializable representation of the object
    """
    # Initialize seen set for circular reference detection
    if _seen is None:
        _seen = set()

    # Check for circular references (only for mutable objects)
    obj_id = id(obj)
    if isinstance(obj, (dict, list, set)) and obj_id in _seen:
        return f"<circular reference: {type(obj).__name__}>"

    # Primitives - return as-is
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Handle PydanticUndefined - convert to None to prevent serialization errors
    # This fixes TypeError in LangGraph checkpoint serialization (msgpack)
    if _is_pydantic_undefined(obj):
        return None

    # Add to seen set for mutable containers
    if isinstance(obj, (dict, list, set)):
        _seen = _seen | {obj_id}  # Create new set to avoid mutation issues

    # Dict - recursively process all values
    if isinstance(obj, dict):
        return {
            _convert_to_serializable(k, _seen): _convert_to_serializable(v, _seen)
            for k, v in obj.items()
        }

    # List/tuple - recursively process all items
    if isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item, _seen) for item in obj]

    # Set - convert to list and process
    if isinstance(obj, set):
        return [_convert_to_serializable(item, _seen) for item in obj]

    # Bytes - decode to string
    if isinstance(obj, bytes):
        try:
            return obj.decode('utf-8')
        except UnicodeDecodeError:
            return obj.decode('utf-8', errors='replace')

    # Datetime objects
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()

    # Pydantic BaseModel (v2) - check for model_dump method
    if hasattr(obj, 'model_dump') and callable(getattr(obj, 'model_dump')):
        try:
            return _convert_to_serializable(obj.model_dump(), _seen)
        except Exception as e:
            logger.debug(f"Failed to call model_dump on {type(obj).__name__}: {e}")

    # Pydantic BaseModel (v1) - check for dict method
    if hasattr(obj, 'dict') and callable(getattr(obj, 'dict')) and hasattr(obj, '__fields__'):
        try:
            return _convert_to_serializable(obj.dict(), _seen)
        except Exception as e:
            logger.debug(f"Failed to call dict on {type(obj).__name__}: {e}")

    # LangChain BaseMessage - extract key fields
    if hasattr(obj, 'type') and hasattr(obj, 'content'):
        try:
            result = {
                "type": obj.type,
                "content": _convert_to_serializable(obj.content, _seen),
            }
            if hasattr(obj, 'additional_kwargs') and obj.additional_kwargs:
                result["additional_kwargs"] = _convert_to_serializable(obj.additional_kwargs, _seen)
            if hasattr(obj, 'name') and obj.name:
                result["name"] = obj.name
            return result
        except Exception as e:
            logger.debug(f"Failed to extract message fields from {type(obj).__name__}: {e}")

    # Objects with __dict__ attribute (custom classes)
    if hasattr(obj, '__dict__'):
        try:
            return _convert_to_serializable(obj.__dict__, _seen)
        except Exception as e:
            logger.debug(f"Failed to serialize __dict__ of {type(obj).__name__}: {e}")

    # UUID objects
    if hasattr(obj, 'hex') and hasattr(obj, 'int'):
        return str(obj)

    # Enum objects
    if hasattr(obj, 'value') and hasattr(obj, 'name') and hasattr(obj.__class__, '__members__'):
        return obj.value

    # Last resort - convert to string
    try:
        return str(obj)
    except Exception:
        return f"<non-serializable: {type(obj).__name__}>"


def patch_langgraph_msgpack_serializer() -> bool:
    """
    Patch LangGraph's msgpack serializer to handle PydanticUndefined.

    This fixes TypeError: Type is not msgpack serializable: PydanticUndefinedType
    that occurs when LangGraph checkpoint system tries to serialize state containing
    Pydantic model fields with undefined default values.

    Should be called once during SDK initialization.

    Returns:
        True if patch was applied successfully, False otherwise
    """
    try:
        from langgraph.checkpoint.serde import jsonplus
        import ormsgpack

        # Store reference to original function
        original_msgpack_default = jsonplus._msgpack_default

        def patched_msgpack_default(obj):
            """
            Enhanced msgpack default serializer that handles PydanticUndefined.
            """
            # Handle PydanticUndefined first
            if _is_pydantic_undefined(obj):
                return None

            # Handle Pydantic FieldInfo objects that may contain PydanticUndefined
            if type(obj).__name__ == 'FieldInfo':
                try:
                    # Convert FieldInfo to a serializable dict, replacing PydanticUndefined
                    field_dict = {}
                    for attr in ['default', 'default_factory', 'description', 'title']:
                        if hasattr(obj, attr):
                            val = getattr(obj, attr)
                            if _is_pydantic_undefined(val):
                                field_dict[attr] = None
                            elif val is not None:
                                field_dict[attr] = str(val) if not isinstance(val, (str, int, float, bool, type(None))) else val
                    return field_dict
                except Exception:
                    return str(obj)

            # Fall back to original function
            return original_msgpack_default(obj)

        # Apply the patch
        jsonplus._msgpack_default = patched_msgpack_default
        logger.debug("Successfully patched LangGraph msgpack serializer for PydanticUndefined handling")
        return True

    except ImportError as e:
        logger.debug(f"LangGraph not available for patching: {e}")
        return False
    except Exception as e:
        logger.warning(f"Failed to patch LangGraph msgpack serializer: {e}")
        return False


# Apply the patch when this module is imported
_langgraph_patch_applied = patch_langgraph_msgpack_serializer()


def safe_serialize(obj: Any, **kwargs) -> str:
    """
    Safely serialize any object to a JSON string.

    Pre-processes the entire object tree to convert non-serializable
    objects before passing to json.dumps. This ensures nested dicts
    and lists with non-standard objects are handled correctly.

    Args:
        obj: Any object to serialize
        **kwargs: Additional arguments passed to json.dumps
            (e.g., indent, sort_keys)

    Returns:
        JSON string representation of the object

    Example:
        >>> from pydantic import BaseModel
        >>> class User(BaseModel):
        ...     name: str
        >>> state = {"user": User(name="Alice"), "count": 5}
        >>> safe_serialize(state)
        '{"user": {"name": "Alice"}, "count": 5}'
    """
    # Pre-process the entire object tree
    serializable = _convert_to_serializable(obj)

    # Set defaults
    kwargs.setdefault('ensure_ascii', False)

    return json.dumps(serializable, **kwargs)

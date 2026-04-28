"""
Regression test for: TypeError: Type is not msgpack serializable: PydanticUndefinedType

Root cause:
  assistant.py:1172 rebuilds swarm tool schemas by copying field metadata:
    _Field(description=field_info.description, default=field_info.default)
  For required fields (no default), field_info.default is PydanticUndefined.
  Pydantic stores that sentinel literally in the new FieldInfo, so the rebuilt
  schema carries PydanticUndefined as a field default. When LangGraph checkpoints
  the pipeline state, _msgpack_enc → ormsgpack.packb hits PydanticUndefined and
  raises TypeError.
"""
from pydantic import Field
from pydantic.fields import Field as _Field
from pydantic import create_model as _create_model
from pydantic_core import PydanticUndefined

import pytest


def _pipeline_state_with_undefined():
    """
    Reproduces the exact leak from assistant.py:1167-1173.

    GitHub toolkit schemas (e.g. BranchName in elitea_sdk/tools/github/schemas.py:34-36)
    have required fields with no default. When assistant.py rebuilds swarm schemas via:
        _Field(description=field_info.description, default=field_info.default)
    field_info.default is PydanticUndefined, which gets stored literally in the new FieldInfo.
    That value then reaches the checkpoint serializer and raises TypeError.
    """
    # Mirrors CreateIssueOnProject from elitea_sdk/tools/github/schemas.py
    original_schema = _create_model(
        "CreateIssueOnProject",
        board_repo=(str, Field(description="Repository slug, e.g. owner/repo")),
        project_title=(str, Field(description="Project title")),
        title=(str, Field(description="Issue title")),
        body=(str, Field(description="Issue body")),
    )

    # Exact code from assistant.py:1167-1173
    swarm_fields = {}
    for field_name, field_info in original_schema.model_fields.items():
        if field_name == "chat_history":
            continue
        swarm_fields[field_name] = (
            field_info.annotation,
            _Field(description=field_info.description, default=field_info.default),  # leaks PydanticUndefined
        )
    rebuilt_schema = _create_model("SwarmSchema", **swarm_fields)

    return {
        "tool_name": "create_issue_on_project",
        "schema_defaults": {
            name: field.default
            for name, field in rebuilt_schema.model_fields.items()
        },
        "result": "The issue with number '42' has been created.",
    }


def test_pydantic_undefined_causes_raw_msgpack_error():
    """
    Proves the bug exists: raw ormsgpack cannot serialize PydanticUndefined.

    This test uses ormsgpack directly (without the LangGraph patch) to verify
    that PydanticUndefined causes TypeError during serialization.
    """
    import ormsgpack

    state = _pipeline_state_with_undefined()

    assert any(
        v is PydanticUndefined for v in state["schema_defaults"].values()
    ), "precondition: PydanticUndefined must be present in state"

    # Raw ormsgpack should fail with TypeError on PydanticUndefined
    with pytest.raises(TypeError, match="PydanticUndefinedType"):
        ormsgpack.packb(state)


def test_pydantic_undefined_in_state_serializes_after_patch():
    """
    Verifies the fix for https://github.com/EliteaAI/elitea_issues/issues/3971

    PydanticUndefined leaks into pipeline state via the swarm schema rebuild in
    assistant.py:1172. Without the patch in elitea_sdk.runtime.utils.serialization,
    this would cause TypeError when LangGraph tries to checkpoint the state via
    msgpack serialization.

    This test verifies that after the patch is applied, serialization succeeds.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from elitea_sdk.runtime.utils.serialization import patch_langgraph_msgpack_serializer

    # Ensure patch is applied
    patch_langgraph_msgpack_serializer()

    serializer = JsonPlusSerializer()
    state = _pipeline_state_with_undefined()

    assert any(
        v is PydanticUndefined for v in state["schema_defaults"].values()
    ), "precondition: PydanticUndefined must be present in state"

    # After the patch, serialization should succeed without raising TypeError
    # The patch converts PydanticUndefined to None during msgpack serialization
    result = serializer.dumps_typed(state)
    assert result is not None, "Serialization should return a result"


def test_extracting_field_defaults_safely():
    """
    Verifies proper handling of field.default when extracting values into state.

    Pydantic always stores PydanticUndefined as .default for required fields,
    even when Field() is called without a default parameter. The fix is to
    check for PydanticUndefined when extracting field defaults into state
    that will be serialized (e.g., LangGraph checkpoints).
    """
    import ormsgpack

    # Create a schema with required and optional fields
    schema = _create_model(
        "ToolSchema",
        required_field=(str, Field(description="A required field")),
        optional_field=(str, Field(description="An optional field", default="default_value")),
    )

    # BAD: directly extracting field.default leaks PydanticUndefined
    bad_state = {
        'schema_defaults': {
            name: field.default
            for name, field in schema.model_fields.items()
        }
    }
    assert any(v is PydanticUndefined for v in bad_state['schema_defaults'].values()), \
        "Precondition: bad extraction should have PydanticUndefined"

    with pytest.raises(TypeError, match="PydanticUndefinedType"):
        ormsgpack.packb(bad_state)

    # GOOD: check for PydanticUndefined and replace with None
    good_state = {
        'schema_defaults': {
            name: None if field.default is PydanticUndefined else field.default
            for name, field in schema.model_fields.items()
        }
    }
    assert not any(v is PydanticUndefined for v in good_state['schema_defaults'].values()), \
        "Fixed extraction should not have PydanticUndefined"

    # Serialization should succeed
    result = ormsgpack.packb(good_state)
    assert result is not None

    # Optional field's default should be preserved
    assert good_state['schema_defaults']['optional_field'] == "default_value"
    # Required field's default should be None (safe placeholder)
    assert good_state['schema_defaults']['required_field'] is None

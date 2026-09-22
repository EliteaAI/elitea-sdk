"""Regression tests for https://github.com/EliteaAI/elitea_issues/issues/6690

MCP tool schemas used to be rebuilt as Pydantic models, which dropped every JSON
Schema keyword the rebuild did not map (bounds, $ref, const, integer enums, type
arrays), dropped tools it could not build (allOf), and stripped arguments allowed
through additionalProperties. The model must see the schema the server declared,
and the server must receive the arguments the model sent.
"""
import contextlib
import copy
import json
import sys
import uuid
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool

from elitea_sdk.runtime.models.mcp_models import McpConnectionConfig, McpToolMetadata
from elitea_sdk.runtime.toolkits.mcp import McpToolkit
from elitea_sdk.runtime.toolkits.tools import _init_single_mcp_tool
from elitea_sdk.runtime.tools.lazy_tools import ToolRegistry
from elitea_sdk.runtime.tools import mcp_input_schema
from elitea_sdk.runtime.tools.mcp_input_schema import build_mcp_args_schema, conform_mcp_arguments
from elitea_sdk.runtime.tools.tool import ToolNode
from tests.runtime.utils.mcp_rig_server import SCHEMA_TOOLS, RunningRig

PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
RECORDED_TOOLS_LIST = Path(__file__).parent / "utils" / "fixtures" / "tools_list_6690.json"
SCHEMA_TOOLS_BY_NAME = {tool.name: tool.inputSchema for tool in SCHEMA_TOOLS}

# What the model sees where it cannot be the declared schema verbatim.
BOUND_PARAMETERS_OVERRIDES = {
    "echo_ref": {
        "type": "object",
        "properties": {"filter": {
            "type": "object",
            "properties": {"state": {"type": "string", "enum": ["open", "closed"]}, "limit": {"type": "integer"}},
            "required": ["state"],
        }},
        "required": ["filter"],
    },
    "echo_ref_network": {"type": "object", "properties": {"doc": {"description": "A document"}}},
    "echo_brackets": {
        "type": "object",
        "properties": {"fname": {"type": "array", "items": {"type": "string"}}},
        "required": ["fname"],
    },
}


def recorded_tools():
    servers = json.loads(RECORDED_TOOLS_LIST.read_text())
    return [
        pytest.param(tool, id=f"{server}-{tool['name']}")
        for server, recording in servers.items()
        for tool in recording["tools"]
    ]


def build_remote_tool(name, input_schema):
    return McpToolkit._create_tool_from_dict(
        tool_dict={"name": name, "description": "d", "inputSchema": input_schema},
        toolkit_name="schemas",
        toolkit_type="mcp",
        connection_config=McpConnectionConfig(url="https://mcp.example.test/mcp"),
        timeout=60,
        client=None,
    )


def build_tool_from_metadata(name, input_schema):
    return McpToolkit._create_tool_from_metadata(
        tool_metadata=McpToolMetadata(name=name, description="d", server="schemas", input_schema=input_schema),
        toolkit_name="schemas",
        toolkit_type="mcp",
        timeout=60,
        client=None,
    )


def build_static_tool(name, input_schema):
    return McpToolkit._create_single_tool(
        toolkit_name="schemas",
        toolkit_type="mcp",
        available_tool={"name": name, "description": "d", "inputSchema": input_schema},
        timeout=60,
        client=None,
    )


def build_proxied_tool(name, input_schema):
    return _init_single_mcp_tool(
        "schemas", "schemas", {"name": name, "description": "d", "inputSchema": input_schema}, None, {}
    )


def bound_parameters(tool):
    return convert_to_openai_tool(tool)["function"]["parameters"]


@pytest.fixture(scope="module")
def rig():
    with RunningRig() as running:
        yield running


@pytest.fixture
def rig_tools(rig, monkeypatch):
    for variable in PROXY_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    toolkit = McpToolkit.get_toolkit(
        url=f"{rig.base_url}/schemas/mcp/",
        toolkit_name="schemas",
        timeout=20,
        session_id=str(uuid.uuid4()),
    )
    return {tool.name: tool for tool in toolkit.get_tools()}


def received_arguments(result):
    return json.loads(result)["received"]


@pytest.mark.parametrize("tool", recorded_tools())
def test_a_recorded_public_server_schema_reaches_the_model_unchanged(tool):
    remote_tool = build_remote_tool(tool["name"], tool["inputSchema"])

    assert bound_parameters(remote_tool) == tool["inputSchema"]


@pytest.mark.parametrize("build_tool", [build_tool_from_metadata, build_static_tool, build_proxied_tool])
def test_every_mcp_tool_builder_binds_the_declared_schema(build_tool):
    input_schema = SCHEMA_TOOLS_BY_NAME["echo_constraints"]

    assert bound_parameters(build_tool("echo_constraints", input_schema)) == input_schema


def test_every_rig_tool_is_discovered(rig_tools):
    assert set(SCHEMA_TOOLS_BY_NAME) <= set(rig_tools)


@pytest.mark.parametrize("name", sorted(SCHEMA_TOOLS_BY_NAME))
def test_a_discovered_tool_binds_with_the_declared_schema(rig_tools, name):
    expected = BOUND_PARAMETERS_OVERRIDES.get(name, SCHEMA_TOOLS_BY_NAME[name])

    assert bound_parameters(rig_tools[name]) == expected


def test_binding_leaves_the_tool_schema_intact(rig_tools):
    tool = rig_tools["echo_ref"]
    declared = copy.deepcopy(tool.args_schema)
    bound_parameters(tool)

    assert tool.args_schema == declared


def test_arguments_allowed_by_additional_properties_reach_the_server(rig_tools):
    result = rig_tools["echo_additional_properties"].invoke({"q": "docs", "lang": "en"})

    assert received_arguments(result) == {"q": "docs", "lang": "en"}


@pytest.mark.parametrize("name, arguments", [
    ("echo_ref", {"filter": {"state": "open", "limit": 5}}),
    ("echo_type_array", {"id": 7}),
    ("echo_int_enum", {"level": 2}),
    ("echo_const", {"mode": "fast"}),
])
def test_arguments_of_a_declared_shape_reach_the_server_unchanged(rig_tools, name, arguments):
    assert received_arguments(rig_tools[name].invoke(arguments)) == arguments


def test_lazy_tool_schema_lookup_returns_the_declared_schema():
    input_schema = SCHEMA_TOOLS_BY_NAME["echo_constraints"]
    registry = ToolRegistry.from_tools([build_remote_tool("echo_constraints", input_schema)])

    assert registry.get_tool_schema("schemas", "echo_constraints")["parameters"] == input_schema


def test_a_property_named_like_a_data_keyword_still_loses_its_external_ref():
    input_schema = {"type": "object", "properties": {
        "default": {"$ref": "https://schemas.example.com/doc.json", "description": "A document"},
    }}

    assert build_mcp_args_schema(input_schema)["args_schema"]["properties"] == {
        "default": {"description": "A document"},
    }


UNBINDABLE_AS_DECLARED = {
    "missing_definition": (
        {"type": "object", "properties": {"f": {"$ref": "#/$defs/Missing", "description": "d"}}},
        {"type": "object", "properties": {"f": {"description": "d"}}},
    ),
    "ref_to_a_renamed_property": (
        {"type": "object", "properties": {"a[]": {"type": "string"}, "b": {"$ref": "#/properties/a[]"}}},
        {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}},
    ),
    "escaped_slash": (
        {"type": "object", "$defs": {"a/b": {"type": "string"}}, "properties": {"f": {"$ref": "#/$defs/a~1b"}}},
        {"type": "object", "properties": {"f": {"type": "string"}}},
    ),
    "percent_encoded": (
        {"type": "object", "$defs": {"a b": {"type": "string"}}, "properties": {"f": {"$ref": "#/$defs/a%20b"}}},
        {"type": "object", "properties": {"f": {"type": "string"}}},
    ),
    "anchor": (
        {"type": "object", "$defs": {"x": {"$anchor": "foo", "type": "string"}}, "properties": {"f": {"$ref": "#foo"}}},
        {"type": "object", "properties": {"f": {}}},
    ),
    "ref_literal_in_default": (
        {"type": "object", "properties": {
            "f": {"type": "object", "default": {"$ref": "#/nope"}}, "g": {"type": "integer", "description": "n"},
        }, "required": ["g"]},
        {"type": "object", "properties": {"f": {"type": "object"}, "g": {"type": "integer", "description": "n"}},
         "required": ["g"]},
    ),
    "nested_property_named_ref": (
        {"type": "object", "properties": {"o": {"type": "object", "properties": {"$ref": {"type": "string"}}}}},
        {"type": "object", "properties": {"o": {"type": "object"}}},
    ),
}


@pytest.mark.parametrize("input_schema, expected", UNBINDABLE_AS_DECLARED.values(), ids=UNBINDABLE_AS_DECLARED)
def test_a_schema_langchain_cannot_resolve_still_binds(input_schema, expected):
    assert bound_parameters(build_remote_tool("t", input_schema)) == expected


def test_one_unbindable_tool_leaves_the_other_tools_typed():
    broken = build_remote_tool("broken", UNBINDABLE_AS_DECLARED["nested_property_named_ref"][0])
    typed = build_remote_tool("typed", SCHEMA_TOOLS_BY_NAME["echo_constraints"])

    assert [bound_parameters(tool) for tool in (broken, typed)][1] == SCHEMA_TOOLS_BY_NAME["echo_constraints"]


def test_string_arguments_are_converted_to_the_declared_scalar_type(rig_tools):
    result = rig_tools["echo_int_enum"].invoke({"level": "2"})

    assert received_arguments(result) == {"level": 2}


def test_arguments_a_closed_schema_forbids_are_dropped(rig_tools):
    result = rig_tools["echo_constraints"].invoke({"query": "docs", "limit": "5", "invented": True})

    assert received_arguments(result) == {"query": "docs", "limit": 5}


@pytest.mark.parametrize("prop, value, expected", [
    ({"type": "number"}, "2.5", 2.5),
    ({"type": "boolean"}, "true", True),
    ({"anyOf": [{"type": "integer"}, {"type": "null"}]}, "3", 3),
    ({"type": ["string", "integer"]}, "8", "8"),
    ({"type": "string"}, "7", "7"),
    ({"type": "integer"}, "five", "five"),
])
def test_scalar_conversion_follows_the_declared_type(prop, value, expected):
    schema = {"type": "object", "properties": {"p": prop}}

    assert conform_mcp_arguments(schema, {"p": value}) == {"p": expected}


class StructuredOutputClient:
    def __init__(self, completion):
        self.completion = completion
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return RunnableLambda(lambda _: self.completion)


def test_a_structured_output_tool_node_runs_an_mcp_tool(rig_tools):
    client = StructuredOutputClient({"level": 2})
    node = ToolNode(client=client, tool=rig_tools["echo_int_enum"], structured_output=True,
                    input_variables=["task"], output_variables=["result"])

    result = RunnableLambda(lambda state, config: node.invoke(state, config=config)).invoke({"task": "level two"})

    assert client.schema["title"] == "NewModel"
    assert client.schema["required"] == ["level"]
    assert received_arguments(result["result"]) == {"level": 2}


def ref_chain(length, branches, last=None):
    definitions = {
        f"D{index}": {"type": "object", "properties": {
            f"p{branch}": {"$ref": f"#/$defs/D{index + 1}"} for branch in range(branches)
        }}
        for index in range(length)
    }
    definitions[f"D{length}"] = last or {"type": "string"}
    return {
        "type": "object",
        "$defs": definitions,
        "properties": {"root": {"$ref": "#/$defs/D0", "description": "entry"}},
        "required": ["root"],
    }


UNTYPED_ROOT = {"type": "object", "properties": {"root": {"description": "entry"}}, "required": ["root"]}


def test_a_recursive_schema_is_cut_where_it_recurses():
    input_schema = {
        "type": "object",
        "$defs": {"Node": {"type": "object", "properties": {"children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}}}},
        "properties": {"tree": {"$ref": "#/$defs/Node"}},
    }

    assert build_mcp_args_schema(input_schema)["args_schema"] == {"type": "object", "properties": {
        "tree": {"type": "object", "properties": {"children": {"type": "array", "items": {}}}},
    }}


def test_a_schema_that_expands_past_the_size_limit_binds_untyped():
    exponential = ref_chain(length=18, branches=2)

    assert build_mcp_args_schema(exponential)["args_schema"] == UNTYPED_ROOT


def test_a_ref_chain_deeper_than_the_stack_still_builds_the_tool():
    tool = build_remote_tool("deep", ref_chain(length=1000, branches=1))

    assert tool is not None
    assert bound_parameters(tool) == UNTYPED_ROOT


def test_a_reused_model_within_the_size_limit_stays_typed():
    shared = ref_chain(length=6, branches=2)

    root = build_mcp_args_schema(shared)["args_schema"]["properties"]["root"]

    assert root["properties"]["p0"] == root["properties"]["p1"]
    assert root["properties"]["p0"]["properties"]["p1"]["type"] == "object"


@pytest.mark.parametrize("last", [
    {"type": "object", "required": [f"key{index}" for index in range(100_000)]},
    {"type": "string", "enum": [f"value{index}" for index in range(100_000)]},
    {"type": "string", "description": "x" * 1_000_000},
    {"type": "object", "properties": {"x" * 1_000_000: {}}},
], ids=["walked_list", "literal_list", "long_string", "long_property_name"])
def test_a_payload_copied_through_a_ref_chain_binds_untyped(last):
    assert build_mcp_args_schema(ref_chain(length=10, branches=2, last=last))["args_schema"] == UNTYPED_ROOT


def test_each_ref_is_resolved_once_per_build(monkeypatch):
    resolved = []
    resolve_pointer = mcp_input_schema._resolve_pointer
    monkeypatch.setattr(mcp_input_schema, "_resolve_pointer",
                        lambda document, ref: resolved.append(ref) or resolve_pointer(document, ref))

    build_mcp_args_schema(ref_chain(length=6, branches=2))

    assert sorted(resolved) == sorted({f"#/$defs/D{index}" for index in range(7)})


FOUR_THOUSAND_DIGIT_INTEGER = int("9" * 4300)

AMPLIFYING_PAYLOADS = {
    "emoji_string": {"type": "string", "description": "😀" * 128},
    "control_characters": {"type": "string", "description": "\x00" * 128},
    "escaped_quotes": {"type": "string", "description": '"' * 128},
    "huge_integer": {"type": "integer", "enum": [FOUR_THOUSAND_DIGIT_INTEGER]},
    "many_huge_integers": {"type": "integer", "enum": [FOUR_THOUSAND_DIGIT_INTEGER] * 32},
    "short_list_items": {"type": "integer", "enum": [0] * 2000},
    "short_walked_items": {"type": "object", "required": ["0"] * 2000},
}


@pytest.mark.parametrize("depth", range(1, 12))
@pytest.mark.parametrize("last", AMPLIFYING_PAYLOADS.values(), ids=AMPLIFYING_PAYLOADS)
def test_inlining_never_grows_the_schema_past_the_allowance(last, depth):
    declared = ref_chain(length=depth, branches=2, last=last)

    bound = build_mcp_args_schema(declared)["args_schema"]

    assert len(json.dumps(bound)) - len(json.dumps(declared)) <= mcp_input_schema._EXPANSION_ALLOWANCE_CHARS


@pytest.mark.parametrize("value", [
    "😀\x00\"\\ plain ü",
    [1.5, float("inf"), -0.0, 1e300, True, False, None],
    {"ü": [{"k": "v"}, []], "": {}},
    [0] * 50,
], ids=["escaped_string", "scalars", "nested", "short_items"])
@pytest.mark.parametrize("estimate", [mcp_input_schema._json_size, mcp_input_schema._walked_json_size],
                         ids=["serialized", "walked"])
def test_the_size_estimate_is_the_serialized_length(estimate, value):
    assert estimate(value) == len(json.dumps(value))


@pytest.mark.parametrize("value", [0, 7, -1, 10, 99, 100, -10**40, 2**64, FOUR_THOUSAND_DIGIT_INTEGER,
                                   FOUR_THOUSAND_DIGIT_INTEGER * 10],
                         ids=["0", "7", "-1", "10", "99", "100", "-1e40", "2**64", "4300_digits", "4301_digits"])
def test_an_integer_size_estimate_is_never_below_its_digits(value):
    with localcontext_without_int_digit_limit():
        digits = len(str(value))

    assert mcp_input_schema._json_size(value) - digits in (0, 1)


@contextlib.contextmanager
def localcontext_without_int_digit_limit():
    limit = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    try:
        yield
    finally:
        sys.set_int_max_str_digits(limit)


def test_a_serializable_value_is_sized_in_one_pass(monkeypatch):
    monkeypatch.setattr(mcp_input_schema, "_walked_json_size", lambda value: pytest.fail("walked a serializable value"))

    assert mcp_input_schema._json_size({"default": [None] * 1000}) == len(json.dumps({"default": [None] * 1000}))


def test_binding_mutates_neither_the_schema_nor_the_servers_document():
    document = {
        "type": "object",
        "$defs": {"Shared": {"type": "object", "default": {"k": [1, {"title": "T"}]}, "examples": [{"title": "y"}],
                             "properties": {"s": {"type": "string", "title": "S"}}}},
        "properties": {"a": {"$ref": "#/$defs/Shared"}, "b": {"$ref": "#/$defs/Shared"}},
    }
    declared = copy.deepcopy(document)
    tool = build_remote_tool("shared", document)
    built = copy.deepcopy(tool.args_schema)

    for _ in range(2):
        bound_parameters(tool)
    ToolNode(tool=tool)._structured_output_schema()

    assert (document, tool.args_schema) == (declared, built)


def test_a_schema_larger_than_the_allowance_stays_typed_when_nothing_expands():
    declared = {"type": "object", "properties": {
        "region": {"type": "string", "enum": [f"region-{index:06d}" for index in range(20_000)]},
    }}
    assert len(json.dumps(declared)) > mcp_input_schema._EXPANSION_ALLOWANCE_CHARS

    assert bound_parameters(build_remote_tool("regions", declared)) == declared

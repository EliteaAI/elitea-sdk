import importlib
import inspect
from typing import Literal, Optional

from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda

import pytest
from pydantic import BaseModel, Field, ValidationError, create_model

from elitea_sdk.runtime.tool_outcome import RUN_FAILURE_NOTICE_KEY
from elitea_sdk.runtime.tools.tool import ToolNode
from elitea_sdk.tools.aha.api_wrapper import AhaCopyRecordInput, AhaManageRecordInput
from elitea_sdk.tools.base.tool import BaseAction

AHA_OUTPUT_FORMAT_TOOLS = (
    "field_options_metadata", "fields_metadata", "find_project", "get_epic", "get_feature", "get_idea",
    "get_initiative", "get_product", "get_release", "get_requirement", "list_comments", "list_epics",
    "list_features", "list_ideas", "list_initiatives", "list_products", "list_releases", "list_requirements",
    "read_records", "search", "search_records",
)
TESTRAIL_OUTPUT_FORMAT_TOOLS = (
    "get_cases", "get_cases_by_filter", "get_results", "get_results_for_case", "get_results_for_run", "get_run",
    "get_runs", "get_sections", "get_suites",
)
MANAGEABLE = ["feature", "requirement", "idea", "release", "initiative", "epic", "page"]
SEARCHABLE = ["feature", "requirement", "release", "idea", "epic", "initiative", "product"]
OUTPUT_FORMATS = ["json", "csv", "markdown"]
FOLDER_TYPES = ["TEST_CASE", "TEST_PLAN", "TEST_CYCLE"]

CLOSED_SET_FIELDS = [
    ("aha", "attach_file", "resource_type", [
        "feature", "requirement", "idea", "release", "release_phase", "epic", "initiative", "goal", "page", "to_do"]),
    *[("aha", tool, "record_type", MANAGEABLE)
      for tool in ("create_record", "update_record", "delete_record", "manage_record")],
    ("aha", "manage_record", "action", ["create", "update", "delete"]),
    ("aha", "search_records", "record_type", SEARCHABLE),
    ("aha", "read_records", "record_type", SEARCHABLE + ["page"]),
    ("aha", "copy_record", "record_type", ["release"]),
    *[("aha", tool, "output_format", OUTPUT_FORMATS) for tool in AHA_OUTPUT_FORMAT_TOOLS],
    *[("testrail", tool, "output_format", OUTPUT_FORMATS) for tool in TESTRAIL_OUTPUT_FORMAT_TOOLS],
    ("confluence", "create_page", "status", ["current", "draft"]),
    ("confluence", "create_pages", "status", ["current", "draft"]),
    ("github", "update_issue", "state", ["open", "closed"]),
    ("github", "get_workflow_runs", "status", [
        "completed", "action_required", "cancelled", "failure", "neutral", "skipped", "stale", "success",
        "timed_out", "in_progress", "queued", "requested", "waiting", "pending"]),
    ("gitlab", "get_issues", "state", ["opened", "closed", "all"]),
    ("gitlab", "list_merge_requests", "state", ["opened", "closed", "locked", "merged", "all"]),
    ("gitlab_org", "get_issues", "state", ["opened", "closed", "all"]),
    ("postman", "analyze", "scope", ["collection", "folder", "request"]),
    ("postman", "get_request_script", "script_type", ["test", "prerequest"]),
    ("confluence", "get_page_id_by_title", "type", ["page", "blogpost"]),
    ("confluence", "add_file_to_page", "position", ["append", "prepend"]),
    ("github", "search_code", "sort", ["indexed"]),
    ("github", "search_code", "order", ["asc", "desc"]),
    ("figma", "extract_design_tokens_batch", "output_format", ["full", "compact", "summary", "style_guide"]),
    ("qtest", "search_entities_by_dql", "object_type", [
        "test-cases", "test-runs", "defects", "requirements", "test-suites", "test-cycles", "releases", "builds",
        "test-logs"]),
    ("report_portal", "get_extended_launch_data_as_raw", "format", ["pdf", "xls", "html"]),
    ("zephyr_essential", "find_folder_by_name", "folder_type", FOLDER_TYPES),
    ("zephyr_essential", "list_folders", "folder_type", FOLDER_TYPES),
    ("zephyr_scale", "get_folders", "folderType", FOLDER_TYPES),
    ("zephyr_scale", "add_test_steps", "tc_mode", ["APPEND", "OVERWRITE"]),
    ("zephyr_scale", "search_test_cases", "order_direction", ["ASC", "DESC"]),
    ("ado", "create_test_case", "test_steps_format", ["json", "xml"]),
    ("ado", "modify_wiki_page", "version_type", ["branch", "tag", "commit"]),
    ("ado", "rename_wiki_page", "version_type", ["branch", "tag", "commit"]),
    ("ado", "get_comments", "expand", ["all", "none", "reactions", "renderedText", "renderedTextOnly"]),
    ("ado", "get_comments", "order", ["asc", "desc"]),
    ("ado", "get_work_item", "expand", ["none", "relations", "fields", "links", "all"]),
]

_args_schemas_cache = {}


def _args_schemas(toolkit: str) -> dict:
    """Tool argument schemas of every toolkit in the package, as the Test Settings form receives them."""
    if toolkit not in _args_schemas_cache:
        module = importlib.import_module(f"elitea_sdk.tools.{toolkit}")
        schemas = {}
        for obj in list(vars(module).values()):
            if inspect.isclass(obj) and "toolkit_config_schema" in vars(obj):
                schema = obj.toolkit_config_schema().model_json_schema()
                schemas.update(schema["properties"]["selected_tools"].get("args_schemas") or {})
        _args_schemas_cache[toolkit] = schemas
    return _args_schemas_cache[toolkit]


def _offered_options(prop: dict) -> Optional[list]:
    """Options the form offers for a property, read the way ToolFormContainer reads them."""
    branches = [branch for branch in prop.get("anyOf", [prop]) if branch.get("type") != "null"]
    if len(branches) != 1:
        return None
    choice = branches[0]
    return choice.get("enum") or ([choice["const"]] if "const" in choice else None)


@pytest.mark.parametrize(("toolkit", "tool", "field", "values"), CLOSED_SET_FIELDS,
                         ids=[f"{t}.{tool}.{f}" for t, tool, f, _ in CLOSED_SET_FIELDS])
def test_closed_set_field_offers_exactly_its_values(toolkit, tool, field, values):
    assert _offered_options(_args_schemas(toolkit)[tool]["properties"][field]) == values


@pytest.mark.parametrize(("toolkit", "tool", "field"), [
    ("aha", "attach_file", "resource_type"),
    ("gitlab", "list_merge_requests", "state"),
    ("report_portal", "get_extended_launch_data_as_raw", "format"),
])
def test_description_names_every_offered_value(toolkit, tool, field):
    prop = _args_schemas(toolkit)[tool]["properties"][field]
    assert [value for value in _offered_options(prop) if value not in prop["description"]] == []


def test_instance_specific_fields_stay_free_text():
    assert _offered_options(_args_schemas("qtest")["update_test_run_status"]["properties"]["status"]) is None
    assert _offered_options(_args_schemas("aha")["search"]["properties"]["type"]) is None


def test_a_value_outside_the_set_is_rejected_with_the_allowed_values():
    with pytest.raises(ValidationError) as error:
        AhaManageRecordInput(action="upsert", record_type="feature")
    assert "'create', 'update' or 'delete'" in str(error.value)


def test_single_value_set_accepts_its_member_only():
    assert AhaCopyRecordInput(record_type="release", record_id="1").record_type == "release"
    with pytest.raises(ValidationError):
        AhaCopyRecordInput(record_type="feature", record_id="1")


class _RecordingWrapper(BaseModel):
    calls: list = []

    def run(self, name, *args, **kwargs):
        self.calls.append((name, kwargs))
        return "ok"


def _action(args_schema):
    return BaseAction(name="get_issues", description="d", args_schema=args_schema, api_wrapper=_RecordingWrapper())


class _StructuredClient:
    """Chat model whose structured output parses a fixed reply into the requested schema."""

    def __init__(self, reply: dict):
        self.reply = reply

    def with_structured_output(self, schema):
        return RunnableLambda(lambda _: schema(**self.reply))


def test_copy_record_points_other_record_types_to_manage_record():
    assert "manage_record" in AhaCopyRecordInput.model_json_schema()["properties"]["record_type"]["description"]


def test_structured_output_outside_the_closed_set_fails_gracefully():
    state = (Optional[Literal["opened", "closed", "all"]], Field(default="opened"))
    action = _action(create_model("A", state=state))
    node = ToolNode(tool=action, client=_StructuredClient({"state": "open"}), structured_output=True,
                    input_variables=["input"])
    reply = node.invoke({"input": "list issues"})
    assert reply["messages"][0]["response_metadata"][RUN_FAILURE_NOTICE_KEY] is True
    assert "with value {'state': 'open'} raised" in reply["messages"][0]["content"]
    assert action.api_wrapper.calls == []


class _ParserWrappingClient(_StructuredClient):
    """Chat model whose json_schema parser wraps the pydantic error in OutputParserException."""

    def with_structured_output(self, schema):
        def parse(_):
            try:
                return schema(**self.reply)
            except ValidationError as error:
                raise OutputParserException("Failed to parse") from error
        return RunnableLambda(parse)


def _state_node(client):
    action = _action(create_model("A", state=(Literal["opened", "closed", "all"], Field())))
    return action, ToolNode(tool=action, client=client, structured_output=True, input_variables=["input"])


def test_parser_wrapped_closed_set_rejection_fails_gracefully():
    action, node = _state_node(_ParserWrappingClient({"state": "open"}))
    reply = node.invoke({"input": "list issues"})
    assert reply["messages"][0]["response_metadata"][RUN_FAILURE_NOTICE_KEY] is True
    assert "with value {'state': 'open'} raised" in reply["messages"][0]["content"]
    assert action.api_wrapper.calls == []


@pytest.mark.parametrize("client_type", [_StructuredClient, _ParserWrappingClient])
def test_other_structured_output_failures_still_propagate(client_type):
    _, node = _state_node(client_type({}))
    with pytest.raises((ValidationError, OutputParserException)):
        node.invoke({"input": "list issues"})

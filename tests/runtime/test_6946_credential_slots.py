"""Credential slots applied in the SDK at the last moment (C-1009 WP-1D slice 1, #6946).

A governed payload carries a slot, never the material. The SDK asks an injected resolver
for the value right before the toolkit is built and writes it, as plain `str`, into a local
copy only. A slot that cannot be resolved stops the run with a typed error; it is never
dropped, because e.g. the GitHub toolkit treats a missing token as anonymous access.
"""

import contextvars
import copy
import json
import pickle
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, ValidationError

import elitea_sdk.tools as sdk_tools
from elitea_sdk.runtime.credential_slots import (
    SLOT_KEY,
    CredentialEnvelope,
    GovernedCredentialError,
    apply_credential_slots,
    has_slots,
    collect_slot_items,
    memoize_slot_scans,
    resolve_run_slots,
    scrub,
)
from elitea_sdk.runtime import credential_slots as cs
from elitea_sdk.runtime.toolkits import tools as runtime_tools
from elitea_sdk.runtime.utils.mcp_oauth import McpContext

TOKEN = "ghp_" + "CANARY" * 6  # token-shaped canary, assembled so no secret-looking literal sits in the file
PEM = "-----BEGIN " + "RSA PRIVATE KEY-----\nCANARYcanary0123456789\nline2line2line2\n-----END " + "RSA PRIVATE KEY-----"
EXPIRES = datetime(2030, 1, 1, tzinfo=timezone.utc)


def slot(fields=("access_token",), generation=1):
    return {
        "v": 1,
        "credential_ref": "53d98fb5-0000-0000-0000-000000000000",
        "credential_generation": generation,
        "decision_id": "d-1",
        "fields": list(fields),
    }


def github_settings(fields=("access_token",), **extra):
    return {
        "github_configuration": {
            "configuration_uuid": "becf88bf-0000-0000-0000-000000000000",
            "base_url": "https://api.github.com",
            SLOT_KEY: slot(fields),
        },
        "selected_tools": ["list_branches"],
        **extra,
    }


class FakeResolver:
    def __init__(self, values=None, generation=1, error=None):
        self.values = {"access_token": TOKEN} if values is None else values
        self.generation = generation
        self.error = error
        self.calls = []

    def __call__(self, *, toolkit_instance_id, field, slot):
        self.calls.append({"toolkit_instance_id": toolkit_instance_id, "field": field, "slot": slot})
        if self.error is not None:
            raise self.error
        return CredentialEnvelope(self.values, self.generation, EXPIRES)


# --- CredentialEnvelope redaction -------------------------------------------------------


def test_envelope_repr_str_format_json_never_show_values():
    env = CredentialEnvelope({"access_token": TOKEN, "a.b": PEM}, 7, EXPIRES)
    for text in (repr(env), str(env), f"{env}", "%s" % env, json.dumps(env, default=str),
                 json.dumps(dict(env.fields), default=str), repr(env.fields)):
        assert TOKEN not in text and "CANARY" not in text and "BEGIN" not in text
    assert repr(env) == "CredentialEnvelope(fields=['a.b', 'access_token'], generation=7)"


def test_envelope_cannot_be_pickled_copied_or_serialised():
    env = CredentialEnvelope({"access_token": TOKEN}, 1)
    for protocol in range(0, pickle.HIGHEST_PROTOCOL + 1):
        with pytest.raises(TypeError):
            pickle.dumps(env, protocol=protocol)
    with pytest.raises(TypeError):
        copy.copy(env)
    with pytest.raises(TypeError):
        copy.deepcopy(env)
    with pytest.raises(TypeError):
        json.dumps(env)
    assert not hasattr(env, "__dict__") and not hasattr(env, "model_dump")
    with pytest.raises(AttributeError):
        env.generation = 2


def test_envelope_rejects_non_string_values():
    with pytest.raises(TypeError):
        CredentialEnvelope({"access_token": 123}, 1)


def test_governed_error_is_typed_fixed_and_picklable():
    err = GovernedCredentialError("UNSUPPORTED_AUTH_MODE")
    assert isinstance(err, ToolException)
    assert err.reason_code == err.code == "UNSUPPORTED_AUTH_MODE"
    assert str(err) == err.message == "This toolkit cannot use a governed credential yet; the run was not started."
    again = pickle.loads(pickle.dumps(err))  # our own error object, round-tripped in-process
    assert again.reason_code == "UNSUPPORTED_AUTH_MODE" and str(again) == str(err)
    # Anything that is not a code never becomes message text.
    odd = GovernedCredentialError(f"boom {TOKEN}")
    assert odd.reason_code == "DEPENDENCY_UNAVAILABLE" and TOKEN not in str(odd)


# --- apply_credential_slots -------------------------------------------------------------


def test_apply_writes_plain_str_into_a_new_dict_and_leaves_inputs_unchanged():
    settings = github_settings(extra_flag={"keep": ["me"]})
    before = copy.deepcopy(settings)
    inner = settings["github_configuration"]
    other = settings["extra_flag"]
    resolver = FakeResolver()

    result, released = apply_credential_slots(settings, "tk-1", resolver)

    assert settings == before                                     # deep equality
    assert settings["github_configuration"] is inner              # identity
    assert SLOT_KEY in inner and "access_token" not in inner
    assert result is not settings
    assert result["github_configuration"] is not inner            # nested dict is new
    assert result["extra_flag"] is other                          # untouched values are shared
    cfg = result["github_configuration"]
    assert SLOT_KEY not in cfg
    assert cfg["access_token"] == TOKEN and type(cfg["access_token"]) is str
    assert cfg["base_url"] == "https://api.github.com"
    assert released == frozenset({TOKEN}) and isinstance(released, frozenset)
    assert resolver.calls == [{"toolkit_instance_id": "tk-1", "field": "github_configuration",
                               "slot": inner[SLOT_KEY]}]


def test_apply_hands_the_resolver_a_copy_of_the_slot():
    settings = github_settings()
    before = copy.deepcopy(settings)

    def mutating(*, toolkit_instance_id, field, slot):
        slot["fields"].append("evil")
        slot["decision_id"] = "x"
        return CredentialEnvelope({"access_token": TOKEN}, 1)

    result, _ = apply_credential_slots(settings, "tk-1", mutating)
    assert settings == before
    assert result["github_configuration"]["access_token"] == TOKEN


def test_apply_writes_nested_paths_without_touching_sibling_nested_dicts():
    settings = {
        "cfg": {
            "auth": {"user": "u", "mode": "x"},
            SLOT_KEY: slot(("auth.password", "token")),
        }
    }
    before = copy.deepcopy(settings)
    nested = settings["cfg"]["auth"]
    resolver = FakeResolver({"auth.password": "pw-CANARY", "token": "tok-CANARY"})

    result, released = apply_credential_slots(settings, "tk", resolver)

    assert settings == before and settings["cfg"]["auth"] is nested
    assert result["cfg"]["auth"] == {"user": "u", "mode": "x", "password": "pw-CANARY"}
    assert result["cfg"]["auth"] is not nested
    assert result["cfg"]["token"] == "tok-CANARY"
    assert released == frozenset({"pw-CANARY", "tok-CANARY"})


def test_apply_creates_missing_parents_for_removed_nested_fields():
    settings = {"cfg": {SLOT_KEY: slot(("auth.password",))}}
    result, _ = apply_credential_slots(settings, "tk", FakeResolver({"auth.password": "pw"}))
    assert result["cfg"] == {"auth": {"password": "pw"}}


def test_apply_without_slots_returns_an_equal_new_dict():
    settings = {"a": {"b": 1}, "c": [1, 2]}
    result, released = apply_credential_slots(settings, "tk", None)
    assert result == settings and result is not settings and released == frozenset()


def test_no_resolver_is_worker_capability_unavailable():
    settings = github_settings()
    before = copy.deepcopy(settings)
    with pytest.raises(GovernedCredentialError) as exc:
        apply_credential_slots(settings, "tk-1", None)
    assert exc.value.reason_code == "WORKER_CAPABILITY_UNAVAILABLE"
    assert settings == before


@pytest.mark.parametrize("code", ["DENIED_GRANT", "STORE_SEALED", "RATE_LIMITED", "CREDENTIAL_REFERENCE_INACTIVE"])
def test_release_refusal_surfaces_its_code(code):
    with pytest.raises(GovernedCredentialError) as exc:
        apply_credential_slots(github_settings(), "tk-1", FakeResolver(error=GovernedCredentialError(code)))
    assert exc.value.reason_code == code


def test_unexpected_resolver_failure_is_typed_and_leaks_nothing():
    resolver = FakeResolver(error=RuntimeError(f"bus exploded for 53d98fb5 {TOKEN}"))
    with pytest.raises(GovernedCredentialError) as exc:
        apply_credential_slots(github_settings(), "tk-1", resolver)
    assert exc.value.reason_code == "DEPENDENCY_UNAVAILABLE"
    assert TOKEN not in str(exc.value) and "53d98fb5" not in str(exc.value)
    assert exc.value.__suppress_context__


def test_a_reply_that_does_not_match_the_slot_is_an_error_never_a_dropped_field():
    with pytest.raises(GovernedCredentialError) as gen:
        apply_credential_slots(github_settings(), "tk-1", FakeResolver(generation=2))
    assert gen.value.reason_code == "CREDENTIAL_GENERATION_MISMATCH"
    with pytest.raises(GovernedCredentialError) as missing:
        apply_credential_slots(github_settings(), "tk-1", FakeResolver(values={}))
    assert missing.value.reason_code == "DEPENDENCY_UNAVAILABLE"
    with pytest.raises(GovernedCredentialError) as extra:
        apply_credential_slots(github_settings(), "tk-1",
                               FakeResolver(values={"access_token": TOKEN, "other": "x"}))
    assert extra.value.reason_code == "DEPENDENCY_UNAVAILABLE"
    with pytest.raises(GovernedCredentialError) as wrong_type:
        apply_credential_slots(github_settings(), "tk-1", lambda **_: {"access_token": TOKEN})
    assert wrong_type.value.reason_code == "DEPENDENCY_UNAVAILABLE"


@pytest.mark.parametrize("bad", [
    None, "x", {"v": 2, "credential_generation": 1, "fields": []},
    {"v": 1, "credential_generation": "1", "fields": []},
    {"v": 1, "credential_generation": 1, "fields": "access_token"},
    {"v": 1, "credential_generation": 1, "fields": ["a..b"]},
])
def test_malformed_slot_is_stale_context(bad):
    with pytest.raises(GovernedCredentialError) as exc:
        apply_credential_slots({"cfg": {SLOT_KEY: bad}}, "tk", FakeResolver())
    assert exc.value.reason_code == "STALE_CONTEXT"


def test_a_slot_that_is_not_directly_under_a_settings_key_is_refused_not_ignored():
    deep = {"outer": {"inner": {SLOT_KEY: slot()}}}
    with pytest.raises(GovernedCredentialError) as exc:
        apply_credential_slots(deep, "tk", FakeResolver())
    assert exc.value.reason_code == "UNSUPPORTED_AUTH_MODE"
    mixed = {"cfg": {"sub": {SLOT_KEY: slot()}, SLOT_KEY: slot()}}
    with pytest.raises(GovernedCredentialError) as exc:
        apply_credential_slots(mixed, "tk", FakeResolver())
    assert exc.value.reason_code == "UNSUPPORTED_AUTH_MODE"


def test_slot_without_fields_needs_no_release():
    settings = {"cfg": {"base_url": "u", SLOT_KEY: slot(())}}
    resolver = FakeResolver()
    result, released = apply_credential_slots(settings, "tk", resolver)
    assert result == {"cfg": {"base_url": "u"}} and released == frozenset() and resolver.calls == []


def test_has_slots_is_a_pure_scan():
    assert has_slots(github_settings())
    assert has_slots({"a": [{"b": {SLOT_KEY: {}}}]})
    assert not has_slots({"a": {"b": 1}, "c": [1, {"d": 2}]})
    assert not has_slots({}) and not has_slots(None) and not has_slots("x")


# --- scrub ------------------------------------------------------------------------------


class _Cfg(BaseModel):
    port: int


def _validation_text(value):
    with pytest.raises(ValidationError) as exc:
        _Cfg(port=value)
    return str(exc.value)


def test_scrub_removes_released_values_from_a_pydantic_validation_error():
    text = _validation_text(TOKEN)
    assert TOKEN in text  # the premise: pydantic echoes the input
    cleaned = scrub(text, frozenset({TOKEN}))
    assert TOKEN not in cleaned and "CANARY" not in cleaned
    assert "[REDACTED]" in cleaned and "Input should be a valid integer" in cleaned


def test_scrub_removes_escaped_and_truncated_forms_of_long_values():
    text = _validation_text(PEM)
    assert "MIIEow" in text or "BEGIN" in text
    cleaned = scrub(text, frozenset({PEM}))
    assert "MIIEow" not in cleaned and "canary" not in cleaned.lower() and "BEGIN RSA" not in cleaned
    long_token = "A" * 30 + "SECRETMIDDLE" + "B" * 30
    cleaned = scrub(_validation_text(long_token), frozenset({long_token}))
    assert "AAAAAAAA" not in cleaned and "BBBBBBBB" not in cleaned
    assert scrub(json.dumps({"k": PEM}), frozenset({PEM})).count("canary") == 0


def test_scrub_is_a_noop_without_values_or_text():
    assert scrub("plain text", frozenset()) == "plain text"
    assert scrub("", frozenset({TOKEN})) == ""
    assert scrub("plain text", frozenset({""})) == "plain text"


# --- tools/__init__.get_tools (SDK built-in toolkits) ------------------------------------


def _tool(name="list_branches"):
    return StructuredTool.from_function(func=lambda: "ok", name=name, description="d")


@pytest.fixture
def recorder(monkeypatch):
    """Replace the github toolkit factory; record the settings it is built from."""
    calls = []

    def get_tools(tool):
        calls.append(tool)
        return [_tool()]

    monkeypatch.setitem(sdk_tools.AVAILABLE_TOOLS, "github", {"get_tools": get_tools})
    return calls


def _github_tool(settings=None, tool_id=42, tool_type="github"):
    return {"type": tool_type, "id": tool_id, "toolkit_name": "gh", "settings": settings or github_settings()}


def test_get_tools_applies_slots_to_a_local_copy_only(recorder):
    tools_list = [_github_tool()]
    before = copy.deepcopy(tools_list)
    resolver = FakeResolver()
    elitea = SimpleNamespace(credential_resolver=resolver)

    result = sdk_tools.get_tools(tools_list, elitea, llm=None)

    assert len(result) == 1
    built_cfg = recorder[0]["settings"]["github_configuration"]
    assert built_cfg["access_token"] == TOKEN and type(built_cfg["access_token"]) is str
    assert SLOT_KEY not in built_cfg
    assert tools_list == before                                    # caller's payload untouched
    assert resolver.calls[0]["toolkit_instance_id"] == "42"
    assert resolver.calls[0]["field"] == "github_configuration"


def test_get_tools_without_resolver_stops_before_any_toolkit_is_built(recorder):
    with pytest.raises(GovernedCredentialError) as exc:
        sdk_tools.get_tools([_github_tool()], SimpleNamespace(), llm=None)
    assert exc.value.reason_code == "WORKER_CAPABILITY_UNAVAILABLE"
    with pytest.raises(GovernedCredentialError):
        sdk_tools.get_tools([_github_tool()], None, llm=None)
    assert recorder == []


def test_github_anonymous_fallback_is_unreachable_when_a_slot_fails(recorder):
    """The GitHub wrapper reads a missing token as '' and runs anonymously; it must not be built."""
    for error in (GovernedCredentialError("DENIED_GRANT"), RuntimeError("bus down")):
        elitea = SimpleNamespace(credential_resolver=FakeResolver(error=error))
        with pytest.raises(GovernedCredentialError):
            sdk_tools.get_tools([_github_tool()], elitea, llm=None)
    elitea = SimpleNamespace(credential_resolver=FakeResolver(values={}))
    with pytest.raises(GovernedCredentialError):
        sdk_tools.get_tools([_github_tool()], elitea, llm=None)
    assert recorder == []


def test_toolkit_errors_are_scrubbed_of_released_values(monkeypatch, caplog):
    def get_tools(tool):
        _Cfg(port=tool["settings"]["github_configuration"]["access_token"])

    monkeypatch.setitem(sdk_tools.AVAILABLE_TOOLS, "github", {"get_tools": get_tools})
    elitea = SimpleNamespace(credential_resolver=FakeResolver())
    with caplog.at_level("DEBUG"):
        with pytest.raises(ToolException) as exc:
            sdk_tools.get_tools([_github_tool()], elitea, llm=None)
    assert "Error getting tools for github" in str(exc.value)
    assert TOKEN not in str(exc.value) and "CANARY" not in str(exc.value)
    assert exc.value.__suppress_context__
    assert TOKEN not in caplog.text and "CANARY" not in caplog.text


def test_toolkit_tool_exceptions_are_scrubbed_too(monkeypatch):
    def get_tools(tool):
        raise ToolException(f"auth failed for {tool['settings']['github_configuration']['access_token']}")

    monkeypatch.setitem(sdk_tools.AVAILABLE_TOOLS, "github", {"get_tools": get_tools})
    elitea = SimpleNamespace(credential_resolver=FakeResolver())
    with pytest.raises(ToolException) as exc:
        sdk_tools.get_tools([_github_tool()], elitea, llm=None)
    assert TOKEN not in str(exc.value) and "[REDACTED]" in str(exc.value)


def test_ado_special_case_errors_are_scrubbed(monkeypatch):
    def get_tools(tool_type, tool):
        raise ValueError(f"bad {tool['settings']['ado_configuration']['personal_access_token']}")

    monkeypatch.setitem(sdk_tools.AVAILABLE_TOOLS, "ado", {"get_tools": get_tools})
    monkeypatch.setitem(sdk_tools.AVAILABLE_TOOLS, "ado_boards", {})
    settings = {"ado_configuration": {SLOT_KEY: slot(("personal_access_token",))}}
    elitea = SimpleNamespace(credential_resolver=FakeResolver({"personal_access_token": TOKEN}))
    with pytest.raises(ToolException) as exc:
        sdk_tools.get_tools([_github_tool(settings, tool_type="ado_boards")], elitea, llm=None)
    assert TOKEN not in str(exc.value)


@pytest.mark.parametrize("tool_type", ["mcp", "planning", "application", "some_community_toolkit"])
def test_get_tools_refuses_slots_on_non_configuration_backed_types(recorder, tool_type):
    resolver = FakeResolver()
    with pytest.raises(GovernedCredentialError) as exc:
        sdk_tools.get_tools([_github_tool(tool_type=tool_type)], SimpleNamespace(credential_resolver=resolver), llm=None)
    assert exc.value.reason_code == "UNSUPPORTED_AUTH_MODE"
    assert resolver.calls == []


def test_get_tools_refuses_slots_on_custom_module_toolkits(recorder):
    settings = github_settings(module="evil.module", **{"class": "Toolkit"})
    resolver = FakeResolver()
    with pytest.raises(GovernedCredentialError) as exc:
        sdk_tools.get_tools([_github_tool(settings, tool_type="custom")], SimpleNamespace(credential_resolver=resolver), llm=None)
    assert exc.value.reason_code == "UNSUPPORTED_AUTH_MODE"
    assert resolver.calls == []


def test_get_tools_is_inert_without_slots(recorder):
    """No slot anywhere: the resolver is never consulted and the built settings are today's."""
    plain = {
        "github_configuration": {"configuration_uuid": "u", "base_url": "https://api.github.com",
                                 "access_token": "legacy-token"},
        "selected_tools": ["list_branches"],
    }
    tools_list = [_github_tool(plain)]
    before = copy.deepcopy(tools_list)
    resolver = FakeResolver()
    elitea = SimpleNamespace(credential_resolver=resolver)
    llm = object()

    sdk_tools.get_tools(tools_list, elitea, llm=llm)

    assert tools_list == before
    assert resolver.calls == []
    built = recorder[0]["settings"]
    assert built == {**plain, "elitea": elitea, "llm": llm, "store": None}
    assert {k: v for k, v in built.items() if k not in ("elitea", "llm", "store")} == before[0]["settings"]
    assert built["github_configuration"] is plain["github_configuration"]   # not even copied
    # Without a client attribute at all, the slotless path is identical.
    recorder.clear()
    sdk_tools.get_tools(tools_list, SimpleNamespace(), llm=llm)
    assert recorder[0]["settings"]["github_configuration"] == plain["github_configuration"]


# --- runtime/toolkits/tools.get_tools (loop head + token map) ------------------------------


@pytest.fixture
def runtime(monkeypatch):
    built = []

    def fake_elitea_tools(tools_list, elitea_client, llm, memory_store):
        built.extend(copy.deepcopy([{k: v for k, v in t.items()} for t in tools_list]))
        return []

    monkeypatch.setattr(runtime_tools, "elitea_tools", fake_elitea_tools)
    monkeypatch.setattr(runtime_tools, "community_tools", lambda *a, **k: [])
    monkeypatch.setattr(runtime_tools, "_mcp_tools", lambda *a, **k: [])
    return built


def _runtime_tool(tool_type, tool_id, settings):
    return {"type": tool_type, "id": tool_id, "toolkit_name": f"{tool_type}-{tool_id}", "name": f"{tool_type}-{tool_id}",
            "settings": settings}


@pytest.mark.parametrize("tool_type", ["mcp", "mcp_config", "application", "some_community_toolkit", "custom"])
@pytest.mark.parametrize("debug_mode", [False, True])
def test_runtime_refuses_a_slot_on_a_tool_type_that_is_not_a_built_in_toolkit(runtime, tool_type, debug_mode):
    tool = _runtime_tool(tool_type, 1, github_settings(module="m", **{"class": "C"}))
    with pytest.raises(GovernedCredentialError) as exc:
        runtime_tools.get_tools([tool], elitea_client=SimpleNamespace(), debug_mode=debug_mode)
    assert exc.value.reason_code == "UNSUPPORTED_AUTH_MODE"
    assert runtime == []


def test_runtime_accepts_slots_on_built_in_toolkits_and_aliases(runtime):
    for tool_type in ("github", "azure_devops_repos", "ado_boards"):
        runtime.clear()
        tool = _runtime_tool(tool_type, 1, github_settings())
        runtime_tools.get_tools([tool], elitea_client=SimpleNamespace())
        assert len(runtime) == 1 and SLOT_KEY in runtime[0]["settings"]["github_configuration"]


def test_tokens_are_not_injected_into_slot_tools_but_still_into_others(runtime):
    tokens = {"https://mcp.example.test": {"access_token": "BROWSER-TOKEN"}}
    slotted = _runtime_tool("github", 1, github_settings())
    plain = _runtime_tool("jira", 2, {"jira_configuration": {"base_url": "https://j"}})
    tools_list = [slotted, plain]

    runtime_tools.get_tools(tools_list, elitea_client=SimpleNamespace(), mcp_context=McpContext(tokens=tokens))

    by_id = {t["id"]: t for t in runtime}
    assert "tokens" not in by_id[1]["settings"]
    assert by_id[2]["settings"]["tokens"] == tokens
    assert "tokens" not in slotted["settings"]


def test_runtime_is_inert_without_slots(runtime):
    tokens = {"u": {"access_token": "T"}}
    plain = _runtime_tool("github", 1, {"github_configuration": {"access_token": "legacy"}})
    expected = copy.deepcopy(plain)
    expected["settings"]["tokens"] = tokens

    runtime_tools.get_tools([plain], elitea_client=SimpleNamespace(), mcp_context=McpContext(tokens=tokens))

    assert runtime == [expected]


# --- batch release and single-scan counts (C-1009 perf PR 5: M1 SDK side, S3, L7) -----------------


def three_slot_settings():
    return {
        "github_configuration": {"base_url": "u", SLOT_KEY: slot(("access_token",))},
        "jira_configuration": {"base_url": "j", SLOT_KEY: slot(("api_key",), generation=3)},
        "pgvector_configuration": {"connection_string": "c", SLOT_KEY: slot(("connection_string",), generation=2)},
        "selected_tools": ["x"],
    }


class BatchResolver:
    """Spy resolver exposing both `resolve_many` and the single-slot call."""

    GENERATIONS = {"github_configuration": 1, "jira_configuration": 3, "pgvector_configuration": 2}
    PATHS = {"github_configuration": "access_token", "jira_configuration": "api_key",
             "pgvector_configuration": "connection_string"}

    def __init__(self, replies=None, error=None):
        self.many_calls = []
        self.single_calls = []
        self.replies = replies
        self.error = error

    def _envelope(self, field):
        return CredentialEnvelope({self.PATHS[field]: f"secret-{field}"}, self.GENERATIONS[field], EXPIRES)

    def resolve_many(self, *, toolkit_instance_id, items):
        self.many_calls.append({"toolkit_instance_id": toolkit_instance_id, "items": items})
        if self.error is not None:
            raise self.error
        if self.replies is not None:
            return self.replies
        return {item["field"]: self._envelope(item["field"]) for item in items}

    def __call__(self, *, toolkit_instance_id, field, slot):
        self.single_calls.append(field)
        return self._envelope(field)


class SingleOnlyResolver:
    def __init__(self):
        self.calls = []

    def __call__(self, *, toolkit_instance_id, field, slot):
        self.calls.append(field)
        return BatchResolver()._envelope(field)


def test_three_slots_are_released_with_one_resolve_many_call():
    settings = three_slot_settings()
    before = copy.deepcopy(settings)
    resolver = BatchResolver()

    result, released = apply_credential_slots(settings, "tk-9", resolver)

    assert len(resolver.many_calls) == 1 and resolver.single_calls == []
    call = resolver.many_calls[0]
    assert call["toolkit_instance_id"] == "tk-9"
    assert [i["field"] for i in call["items"]] == ["github_configuration", "jira_configuration", "pgvector_configuration"]
    assert call["items"][1]["slot"] == settings["jira_configuration"][SLOT_KEY]
    assert call["items"][1]["slot"] is not settings["jira_configuration"][SLOT_KEY]   # a copy
    assert result["github_configuration"]["access_token"] == "secret-github_configuration"
    assert result["jira_configuration"]["api_key"] == "secret-jira_configuration"
    assert result["pgvector_configuration"]["connection_string"] == "secret-pgvector_configuration"
    assert all(SLOT_KEY not in result[k] for k in BatchResolver.PATHS)
    assert released == frozenset(f"secret-{k}" for k in BatchResolver.PATHS)
    assert settings == before


def test_resolver_without_resolve_many_gets_three_single_calls():
    resolver = SingleOnlyResolver()
    result, released = apply_credential_slots(three_slot_settings(), "tk-9", resolver)
    assert resolver.calls == ["github_configuration", "jira_configuration", "pgvector_configuration"]
    assert len(released) == 3 and result["jira_configuration"]["api_key"] == "secret-jira_configuration"


def test_slots_without_fields_do_not_reach_the_resolver_at_all():
    resolver = BatchResolver()
    apply_credential_slots({"cfg": {SLOT_KEY: slot(())}}, "tk", resolver)
    assert resolver.many_calls == [] and resolver.single_calls == []


def test_a_malformed_second_slot_is_refused_before_any_release():
    settings = three_slot_settings()
    settings["jira_configuration"][SLOT_KEY] = {"v": 2, "credential_generation": 1, "fields": ["api_key"]}
    resolver = BatchResolver()
    with pytest.raises(GovernedCredentialError) as exc:
        apply_credential_slots(settings, "tk", resolver)
    assert exc.value.reason_code == "STALE_CONTEXT"
    assert resolver.many_calls == [] and resolver.single_calls == []


def _batch_envelope(field, generation=None, paths=None):
    return CredentialEnvelope({p: "v" for p in (paths or [BatchResolver.PATHS[field]])},
                              generation if generation is not None else BatchResolver.GENERATIONS[field])


def test_every_batch_envelope_is_validated_against_its_own_slot():
    good = {f: _batch_envelope(f) for f in BatchResolver.PATHS}
    cases = {
        "generation": ({**good, "jira_configuration": _batch_envelope("jira_configuration", generation=9)},
                       "CREDENTIAL_GENERATION_MISMATCH"),
        "swapped": ({**good, "github_configuration": good["jira_configuration"],
                     "jira_configuration": good["github_configuration"]}, "CREDENTIAL_GENERATION_MISMATCH"),
        "field set": ({**good, "pgvector_configuration": _batch_envelope(
            "pgvector_configuration", paths=["connection_string", "extra"])}, "DEPENDENCY_UNAVAILABLE"),
        "missing field": ({k: v for k, v in good.items() if k != "jira_configuration"}, "DEPENDENCY_UNAVAILABLE"),
        "unknown field": ({**good, "other_configuration": good["github_configuration"]}, "DEPENDENCY_UNAVAILABLE"),
        "not an envelope": ({**good, "github_configuration": {"access_token": TOKEN}}, "DEPENDENCY_UNAVAILABLE"),
        "not a mapping": ([good["github_configuration"]], "DEPENDENCY_UNAVAILABLE"),
    }
    for name, (replies, code) in cases.items():
        with pytest.raises(GovernedCredentialError) as exc:
            apply_credential_slots(three_slot_settings(), "tk", BatchResolver(replies=replies))
        assert exc.value.reason_code == code, name


def test_resolve_many_failures_are_typed_and_leak_nothing():
    with pytest.raises(GovernedCredentialError) as refused:
        apply_credential_slots(three_slot_settings(), "tk", BatchResolver(error=GovernedCredentialError("DENIED_GRANT")))
    assert refused.value.reason_code == "DENIED_GRANT"
    with pytest.raises(GovernedCredentialError) as boom:
        apply_credential_slots(three_slot_settings(), "tk",
                               BatchResolver(error=RuntimeError(f"bus exploded {TOKEN} 53d98fb5")))
    assert boom.value.reason_code == "DEPENDENCY_UNAVAILABLE"
    assert TOKEN not in str(boom.value) and "53d98fb5" not in str(boom.value) and boom.value.__suppress_context__


def test_a_slot_inside_a_list_or_at_the_root_is_refused_not_ignored():
    for settings in (
        {"cfg": {SLOT_KEY: slot()}, "other": [{"x": {SLOT_KEY: slot()}}]},
        {"cfg": {SLOT_KEY: slot()}, SLOT_KEY: slot()},
    ):
        resolver = BatchResolver()
        with pytest.raises(GovernedCredentialError) as exc:
            apply_credential_slots(settings, "tk", resolver)
        assert exc.value.reason_code == "UNSUPPORTED_AUTH_MODE"
        assert resolver.many_calls == []


@pytest.fixture
def walks(monkeypatch):
    """Count full traversals of a settings dict."""
    calls = []
    real = cs._scan_settings

    def spy(settings):
        calls.append(settings)
        return real(settings)

    monkeypatch.setattr(cs, "_scan_settings", spy)
    return calls


def test_scans_are_memoized_only_inside_a_scope(walks):
    settings = three_slot_settings()
    has_slots(settings), has_slots(settings)
    assert len(walks) == 2                                   # no scope: today's behaviour, nothing cached

    walks.clear()

    @memoize_slot_scans
    def scoped():
        return [has_slots(settings) for _ in range(5)] + [has_slots(three_slot_settings())]

    assert scoped() == [True] * 6
    assert len(walks) == 2                                   # one per distinct dict
    has_slots(settings)
    assert len(walks) == 3                                   # the scope ended; the cache is gone


def test_get_tools_builds_three_slots_with_one_resolver_call_and_one_walk(recorder, walks):
    resolver = BatchResolver()
    elitea = SimpleNamespace(credential_resolver=resolver)
    tools_list = [{"type": "github", "id": 7, "toolkit_name": "gh", "settings": three_slot_settings()}]

    sdk_tools.get_tools(tools_list, elitea, llm=None)

    assert len(resolver.many_calls) == 1 and resolver.single_calls == []
    assert len(walks) == 1
    built = recorder[0]["settings"]
    assert built["jira_configuration"]["api_key"] == "secret-jira_configuration"
    assert all(SLOT_KEY not in built[k] for k in BatchResolver.PATHS)


def test_get_tools_fallback_resolver_gets_one_call_per_slot(recorder):
    resolver = SingleOnlyResolver()
    elitea = SimpleNamespace(credential_resolver=resolver)
    tools_list = [{"type": "github", "id": 7, "toolkit_name": "gh", "settings": three_slot_settings()}]
    sdk_tools.get_tools(tools_list, elitea, llm=None)
    assert resolver.calls == ["github_configuration", "jira_configuration", "pgvector_configuration"]


def test_slotless_settings_cost_one_walk_and_no_resolver_call(recorder, walks, monkeypatch):
    plain = {"github_configuration": {"base_url": "u", "access_token": "legacy"}, "selected_tools": ["list_branches"]}
    resolver = BatchResolver()
    elitea_client = SimpleNamespace(credential_resolver=resolver)
    monkeypatch.setattr(runtime_tools, "community_tools", lambda *a, **k: [])
    monkeypatch.setattr(runtime_tools, "_mcp_tools", lambda *a, **k: [])
    tokens = {"u": {"access_token": "T"}}

    # Whole runtime path: loop-head gate, mcp token branch, then the SDK get_tools gate.
    runtime_tools.get_tools([_runtime_tool("github", 1, plain)], elitea_client=elitea_client,
                            mcp_context=McpContext(tokens=tokens))

    assert len(walks) == 1
    assert resolver.many_calls == [] and resolver.single_calls == []
    assert recorder[0]["settings"]["github_configuration"] == plain["github_configuration"]

    walks.clear()
    sdk_tools.get_tools([_github_tool(plain)], elitea_client, llm=None)   # direct SDK entry: its own scope
    assert len(walks) == 1


def test_governed_runtime_build_walks_once_and_releases_once(recorder, walks, monkeypatch):
    resolver = BatchResolver()
    elitea_client = SimpleNamespace(credential_resolver=resolver)
    monkeypatch.setattr(runtime_tools, "community_tools", lambda *a, **k: [])
    monkeypatch.setattr(runtime_tools, "_mcp_tools", lambda *a, **k: [])

    runtime_tools.get_tools([_runtime_tool("github", 1, three_slot_settings())], elitea_client=elitea_client,
                            mcp_context=McpContext(tokens={"u": {"access_token": "T"}}))

    assert len(walks) == 1
    assert len(resolver.many_calls) == 1 and resolver.single_calls == []
    assert "tokens" not in recorder[0]["settings"]


# --- review fixes: untrusted caller scan, bounded memo scope, depth boundary ----------------------


def test_a_caller_supplied_scan_is_checked_against_the_settings():
    settings = three_slot_settings()
    good = cs.scan_slots(settings)
    apply_credential_slots(settings, "tk", BatchResolver(), scan=good)              # the real scan is accepted

    stale = [
        cs.SlotScan(False, (), False),                                              # claims slotless
        cs.SlotScan(True, good.slotted[:2], False),                                 # misses a slotted key
        cs.SlotScan(True, good.slotted + ("selected_tools",), False),               # names a key with no slot
        cs.SlotScan(True, ("missing_configuration",) + good.slotted, False),        # names an absent key
        cs.SlotScan(True, tuple(reversed(good.slotted)), False),
    ]
    for scan in stale:
        resolver = BatchResolver()
        with pytest.raises(GovernedCredentialError) as exc:
            apply_credential_slots(copy.deepcopy(settings), "tk", resolver, scan=scan)
        assert exc.value.reason_code == "STALE_CONTEXT"
        assert resolver.many_calls == []
    # A root-level slot with a "slotless" scan is refused too.
    with pytest.raises(GovernedCredentialError) as exc:
        apply_credential_slots({SLOT_KEY: slot()}, "tk", BatchResolver(), scan=cs.SlotScan(False, (), False))
    assert exc.value.reason_code == "STALE_CONTEXT"


def test_scope_is_reset_after_a_raise_and_nested_calls_share_the_outer_scope(walks):
    settings = three_slot_settings()

    @memoize_slot_scans
    def inner():
        return has_slots(settings)

    @memoize_slot_scans
    def outer():
        return [has_slots(settings), inner(), inner()]

    assert outer() == [True, True, True]
    assert len(walks) == 1                                   # nested calls reused the outer scope

    @memoize_slot_scans
    def boom():
        has_slots(settings)
        raise RuntimeError("x")

    walks.clear()
    with pytest.raises(RuntimeError):
        boom()
    assert cs._SCAN_SCOPE.get() is None                      # reset in finally
    has_slots(settings)
    has_slots(settings)
    assert len(walks) == 3                                   # 1 inside boom, then 2 uncached


def test_a_copied_context_that_outlives_the_scope_does_not_serve_cached_scans(walks):
    settings = three_slot_settings()
    captured = []

    @memoize_slot_scans
    def scoped():
        has_slots(settings)
        captured.append(contextvars.copy_context())

    scoped()
    walks.clear()
    captured[0].run(lambda: (has_slots(settings), has_slots(settings)))
    assert len(walks) == 2                                   # the closed scope is ignored, nothing cached


def _slot_at_level(level, wrapper_key):
    node = {SLOT_KEY: slot()}
    for _ in range(level):
        node = {wrapper_key: node}
    return node


def test_scan_depth_boundary_is_level_7_on_both_paths():
    # Root = level 0. Non-slotted top-level value is level 1; a slotted config's children are level 2.
    for level in (2, 7):
        plain_path = {"other": _slot_at_level(level - 1, "a")}
        slotted_path = {"cfg": {SLOT_KEY: slot(), "sub": _slot_at_level(level - 2, "a")}}
        assert cs.scan_slots(plain_path).nested and has_slots(plain_path), level
        assert cs.scan_slots(slotted_path).nested, level
    beyond_plain = {"other": _slot_at_level(7, "a")}
    beyond_slotted = {"cfg": {SLOT_KEY: slot(), "sub": _slot_at_level(6, "a")}}
    assert not cs.scan_slots(beyond_plain).found
    assert not cs.scan_slots(beyond_slotted).nested and cs.scan_slots(beyond_slotted).slotted == ("cfg",)


# --- run level: the slots of every toolkit of a run in ONE resolver call (perf PR 5) ------------------


class RunResolver:
    """Spy resolver exposing `resolve_all`, `resolve_many` and the single-slot call."""

    def __init__(self, error=None, generation=1, omit=None):
        self.all_calls, self.many_calls, self.single_calls = [], [], []
        self.error, self.generation, self.omit = error, generation, omit

    def _envelope(self, toolkit_instance_id):
        return CredentialEnvelope({"access_token": f"secret-{toolkit_instance_id}"}, self.generation, EXPIRES)

    def resolve_all(self, *, items):
        self.all_calls.append(items)
        if self.error is not None:
            raise self.error
        replies = {(i["toolkit_instance_id"], i["field"]): self._envelope(i["toolkit_instance_id"]) for i in items}
        replies.pop(self.omit, None)
        return replies

    def resolve_many(self, *, toolkit_instance_id, items):
        self.many_calls.append(toolkit_instance_id)
        return {i["field"]: self._envelope(toolkit_instance_id) for i in items}

    def __call__(self, *, toolkit_instance_id, field, slot):
        self.single_calls.append(toolkit_instance_id)
        return self._envelope(toolkit_instance_id)


def _six_tools():
    return [_github_tool(tool_id=n) for n in range(1, 7)]


def _built_tokens(recorder):
    return [t["settings"]["github_configuration"]["access_token"] for t in recorder]


def test_six_toolkits_are_released_with_one_resolve_all_call(recorder):
    resolver = RunResolver()
    tools_list = _six_tools()
    before = copy.deepcopy(tools_list)

    result = sdk_tools.get_tools(tools_list, SimpleNamespace(credential_resolver=resolver), llm=None)

    assert len(result) == 6 and len(resolver.all_calls) == 1
    assert resolver.many_calls == [] and resolver.single_calls == []
    assert [(i["toolkit_instance_id"], i["field"]) for i in resolver.all_calls[0]] == [
        (str(n), "github_configuration") for n in range(1, 7)]
    assert resolver.all_calls[0][2]["slot"] == tools_list[2]["settings"]["github_configuration"][SLOT_KEY]
    assert _built_tokens(recorder) == [f"secret-{n}" for n in range(1, 7)]
    assert all(SLOT_KEY not in t["settings"]["github_configuration"] for t in recorder)
    assert tools_list == before                                    # caller's payload untouched


def test_toolkits_without_settings_or_slots_do_not_reach_the_resolver(recorder):
    plain = {"github_configuration": {"base_url": "u", "access_token": "legacy"}, "selected_tools": ["list_branches"]}
    tools_list = [_github_tool(tool_id=1), {"type": "github", "id": 2, "settings": {}}, _github_tool(plain, tool_id=3),
                  _github_tool(tool_id=4)]
    resolver = RunResolver()
    sdk_tools.get_tools(tools_list, SimpleNamespace(credential_resolver=resolver), llm=None)
    assert [i["toolkit_instance_id"] for i in resolver.all_calls[0]] == ["1", "4"] and len(resolver.all_calls) == 1
    assert _built_tokens(recorder) == ["secret-1", "legacy", "secret-4"]


def test_a_run_without_slots_makes_no_resolve_all_call(recorder):
    plain = {"github_configuration": {"base_url": "u", "access_token": "legacy"}, "selected_tools": ["list_branches"]}
    resolver = RunResolver()
    sdk_tools.get_tools([_github_tool(plain, tool_id=n) for n in (1, 2)], SimpleNamespace(credential_resolver=resolver), llm=None)
    assert resolver.all_calls == [] and resolver.many_calls == [] and resolver.single_calls == []


def test_a_resolver_with_only_resolve_many_keeps_one_call_per_toolkit(recorder):
    class ManyOnly:
        def __init__(self):
            self.many_calls = []

        def resolve_many(self, *, toolkit_instance_id, items):
            self.many_calls.append(toolkit_instance_id)
            return {i["field"]: CredentialEnvelope({"access_token": f"secret-{toolkit_instance_id}"}, 1, EXPIRES)
                    for i in items}

    resolver = ManyOnly()
    sdk_tools.get_tools(_six_tools(), SimpleNamespace(credential_resolver=resolver), llm=None)
    assert resolver.many_calls == [str(n) for n in range(1, 7)]
    assert _built_tokens(recorder) == [f"secret-{n}" for n in range(1, 7)]


def test_a_single_slot_resolver_keeps_one_call_per_slot(recorder):
    resolver = FakeResolver()
    sdk_tools.get_tools(_six_tools(), SimpleNamespace(credential_resolver=resolver), llm=None)
    assert [c["toolkit_instance_id"] for c in resolver.calls] == [str(n) for n in range(1, 7)]
    assert len(recorder) == 6


def test_a_refused_slot_aborts_the_run_before_the_first_toolkit_is_built(recorder):
    resolver = RunResolver(error=GovernedCredentialError("DENIED_GRANT"))
    with pytest.raises(GovernedCredentialError) as exc:
        sdk_tools.get_tools(_six_tools(), SimpleNamespace(credential_resolver=resolver), llm=None)
    assert exc.value.reason_code == "DENIED_GRANT" and recorder == [] and len(resolver.all_calls) == 1


def test_an_unexpected_resolver_failure_is_typed_and_leaks_nothing(recorder):
    resolver = RunResolver(error=RuntimeError(TOKEN))
    with pytest.raises(GovernedCredentialError) as exc:
        sdk_tools.get_tools(_six_tools(), SimpleNamespace(credential_resolver=resolver), llm=None)
    assert exc.value.reason_code == "DEPENDENCY_UNAVAILABLE" and TOKEN not in str(exc.value)
    assert exc.value.__cause__ is None and recorder == []


def test_a_reply_missing_the_third_toolkit_aborts_before_any_build(recorder):
    resolver = RunResolver(omit=("3", "github_configuration"))
    with pytest.raises(GovernedCredentialError) as exc:
        sdk_tools.get_tools(_six_tools(), SimpleNamespace(credential_resolver=resolver), llm=None)
    assert exc.value.reason_code == "DEPENDENCY_UNAVAILABLE" and recorder == []


def test_a_stale_envelope_of_the_third_toolkit_aborts_before_the_first_is_built(recorder):
    class StaleThird(RunResolver):
        def _envelope(self, toolkit_instance_id):
            generation = 9 if toolkit_instance_id == "3" else 1
            return CredentialEnvelope({"access_token": f"secret-{toolkit_instance_id}"}, generation, EXPIRES)

    with pytest.raises(GovernedCredentialError) as exc:
        sdk_tools.get_tools(_six_tools(), SimpleNamespace(credential_resolver=StaleThird()), llm=None)
    assert exc.value.reason_code == "CREDENTIAL_GENERATION_MISMATCH" and recorder == []


def test_the_scan_walks_each_tool_once_with_the_run_level_call(recorder, walks):
    resolver = RunResolver()
    sdk_tools.get_tools(_six_tools(), SimpleNamespace(credential_resolver=resolver), llm=None)
    assert len(walks) == 6 and len(resolver.all_calls) == 1


def test_an_unsupported_type_is_refused_before_the_batch_call(recorder):
    resolver = RunResolver()
    tools_list = [_github_tool(tool_id=1), _github_tool(tool_id=2, tool_type="some_community_toolkit"),
                  _github_tool(tool_id=3)]
    with pytest.raises(GovernedCredentialError) as exc:
        sdk_tools.get_tools(tools_list, SimpleNamespace(credential_resolver=resolver), llm=None)
    assert exc.value.reason_code == "UNSUPPORTED_AUTH_MODE"
    assert resolver.all_calls == [] and resolver.many_calls == [] and recorder == []


def test_a_nested_or_malformed_slot_is_refused_before_the_batch_call(recorder):
    nested = github_settings()
    nested["github_configuration"]["sub"] = {SLOT_KEY: slot()}
    malformed = github_settings()
    malformed["github_configuration"][SLOT_KEY] = {"v": 2, "credential_generation": 1, "fields": ["access_token"]}
    for bad, code in ((nested, "UNSUPPORTED_AUTH_MODE"), (malformed, "STALE_CONTEXT")):
        resolver = RunResolver()
        tools_list = [_github_tool(tool_id=1), _github_tool(bad, tool_id=2)]
        with pytest.raises(GovernedCredentialError) as exc:
            sdk_tools.get_tools(tools_list, SimpleNamespace(credential_resolver=resolver), llm=None)
        assert exc.value.reason_code == code and resolver.all_calls == [] and recorder == []


def test_a_repeated_toolkit_id_falls_back_to_one_call_per_toolkit(recorder):
    resolver = RunResolver()
    sdk_tools.get_tools([_github_tool(tool_id=7), _github_tool(tool_id=7)], SimpleNamespace(credential_resolver=resolver), llm=None)
    assert resolver.all_calls == [] and resolver.many_calls == ["7", "7"] and len(recorder) == 2


def test_toolkit_errors_stay_scrubbed_with_run_level_envelopes(recorder, monkeypatch):
    def failing(tool):
        raise ValueError(f"bad token {tool['settings']['github_configuration']['access_token']}")

    monkeypatch.setitem(sdk_tools.AVAILABLE_TOOLS, "github", {"get_tools": failing})
    with pytest.raises(ToolException) as exc:
        sdk_tools.get_tools(_six_tools(), SimpleNamespace(credential_resolver=RunResolver()), llm=None)
    assert "secret-1" not in str(exc.value) and cs.REDACTED in str(exc.value)


def test_released_slots_are_applied_without_a_resolver_and_returned_for_scrubbing():
    settings = three_slot_settings()
    released = {k: BatchResolver()._envelope(k) for k in BatchResolver.PATHS}
    result, values = apply_credential_slots(settings, "tk-9", None, released=released)
    assert result["jira_configuration"]["api_key"] == "secret-jira_configuration"
    assert values == frozenset(f"secret-{k}" for k in BatchResolver.PATHS)
    assert all(SLOT_KEY not in result[k] for k in BatchResolver.PATHS) and SLOT_KEY in settings["github_configuration"]


def test_released_envelopes_are_validated_like_fresh_replies():
    settings = three_slot_settings()
    good = {k: BatchResolver()._envelope(k) for k in BatchResolver.PATHS}
    cases = [
        ({k: v for k, v in good.items() if k != "jira_configuration"}, "DEPENDENCY_UNAVAILABLE"),
        ({**good, "extra": good["jira_configuration"]}, "DEPENDENCY_UNAVAILABLE"),
        ({**good, "jira_configuration": "not-an-envelope"}, "DEPENDENCY_UNAVAILABLE"),
        ({**good, "jira_configuration": CredentialEnvelope({"api_key": "x"}, 99, EXPIRES)}, "CREDENTIAL_GENERATION_MISMATCH"),
        ({**good, "jira_configuration": CredentialEnvelope({"other": "x"}, 3, EXPIRES)}, "DEPENDENCY_UNAVAILABLE"),
        ("nope", "DEPENDENCY_UNAVAILABLE"),
    ]
    for released, code in cases:
        with pytest.raises(GovernedCredentialError) as exc:
            apply_credential_slots(settings, "tk", None, released=released)
        assert exc.value.reason_code == code


def test_released_slots_never_call_the_resolver():
    resolver = BatchResolver()
    released = {k: resolver._envelope(k) for k in BatchResolver.PATHS}
    apply_credential_slots(three_slot_settings(), "tk", resolver, released=released)
    assert resolver.many_calls == [] and resolver.single_calls == []


def test_collect_slot_items_names_the_toolkit_and_validates_like_apply():
    items = collect_slot_items(three_slot_settings(), "tk-9")
    assert [(i["toolkit_instance_id"], i["field"]) for i in items] == [
        ("tk-9", "github_configuration"), ("tk-9", "jira_configuration"), ("tk-9", "pgvector_configuration")]
    assert collect_slot_items({"cfg": {"a": 1}}, "tk") == [] and collect_slot_items({"cfg": {SLOT_KEY: slot(())}}, "tk") == []
    with pytest.raises(GovernedCredentialError) as exc:
        collect_slot_items(three_slot_settings(), "")
    assert exc.value.reason_code == "STALE_CONTEXT"


def test_resolve_run_slots_groups_the_reply_by_toolkit_and_checks_its_keys():
    resolver = RunResolver()
    items = [{"toolkit_instance_id": t, "field": "github_configuration", "slot": slot()} for t in ("a", "b")]
    assert {t: set(fields) for t, fields in resolve_run_slots(resolver, items).items()} == {
        "a": {"github_configuration"}, "b": {"github_configuration"}}
    resolver.omit = ("b", "github_configuration")
    with pytest.raises(GovernedCredentialError) as exc:
        resolve_run_slots(resolver, items)
    assert exc.value.reason_code == "DEPENDENCY_UNAVAILABLE"


def test_the_runtime_build_releases_every_built_in_toolkit_in_one_call(recorder, walks, monkeypatch):
    resolver = RunResolver()
    elitea_client = SimpleNamespace(credential_resolver=resolver)
    monkeypatch.setattr(runtime_tools, "community_tools", lambda *a, **k: [])
    monkeypatch.setattr(runtime_tools, "_mcp_tools", lambda *a, **k: [])
    tools_list = [_runtime_tool("github", n, github_settings()) for n in range(1, 7)]

    result = runtime_tools.get_tools(tools_list, elitea_client=elitea_client, mcp_context=McpContext(tokens={"u": {"access_token": "T"}}))

    assert len(result) == 6 and len(resolver.all_calls) == 1 and resolver.many_calls == [] and resolver.single_calls == []
    assert _built_tokens(recorder) == [f"secret-{n}" for n in range(1, 7)]
    assert len(walks) == 6
    assert all("tokens" not in t["settings"] for t in recorder)


def test_the_runtime_build_aborts_before_the_first_built_in_toolkit_when_a_slot_is_refused(recorder, monkeypatch):
    resolver = RunResolver(error=GovernedCredentialError("DENIED_RBAC"))
    monkeypatch.setattr(runtime_tools, "community_tools", lambda *a, **k: [])
    monkeypatch.setattr(runtime_tools, "_mcp_tools", lambda *a, **k: [])
    tools_list = [_runtime_tool("github", n, github_settings()) for n in range(1, 4)]
    with pytest.raises(GovernedCredentialError) as exc:
        runtime_tools.get_tools(tools_list, elitea_client=SimpleNamespace(credential_resolver=resolver))
    assert exc.value.reason_code == "DENIED_RBAC" and recorder == []


# --- review fixes: expiry of up-front releases, shared pre-check, id contract, release lifetime ---------

PAST = datetime(2020, 1, 1, tzinfo=timezone.utc)


def test_an_expired_released_envelope_is_stale_context_but_a_fresh_reply_is_not_rechecked():
    settings = three_slot_settings()
    released = {k: BatchResolver()._envelope(k) for k in BatchResolver.PATHS}
    released["jira_configuration"] = CredentialEnvelope({"api_key": "x"}, 3, PAST)
    with pytest.raises(GovernedCredentialError) as exc:
        apply_credential_slots(settings, "tk", None, released=released)
    assert exc.value.reason_code == "STALE_CONTEXT"
    naive_past = CredentialEnvelope({"api_key": "x"}, 3, PAST.replace(tzinfo=None))
    with pytest.raises(GovernedCredentialError):
        apply_credential_slots(settings, "tk", None, released={**released, "jira_configuration": naive_past})
    no_expiry = CredentialEnvelope({"api_key": "x"}, 3, None)
    apply_credential_slots(settings, "tk", None, released={**released, "jira_configuration": no_expiry})


def test_an_expired_envelope_of_the_third_toolkit_aborts_the_run_before_any_build(recorder):
    class ExpiredThird(RunResolver):
        def _envelope(self, toolkit_instance_id):
            expires = PAST if toolkit_instance_id == "3" else EXPIRES
            return CredentialEnvelope({"access_token": f"secret-{toolkit_instance_id}"}, 1, expires)

    with pytest.raises(GovernedCredentialError) as exc:
        sdk_tools.get_tools(_six_tools(), SimpleNamespace(credential_resolver=ExpiredThird()), llm=None)
    assert exc.value.reason_code == "STALE_CONTEXT" and recorder == []


def test_invalid_tool_names_of_a_later_toolkit_stop_the_run_before_any_release(recorder):
    bad = github_settings()
    bad["selected_tools"] = ["_x"]
    resolver = RunResolver()
    tools_list = [_github_tool(tool_id=n) for n in (1, 2, 3)] + [_github_tool(bad, tool_id=4)] + [
        _github_tool(tool_id=n) for n in (5, 6)]
    with pytest.raises(ValueError):
        sdk_tools.get_tools(tools_list, SimpleNamespace(credential_resolver=resolver), llm=None)
    assert resolver.all_calls == [] and resolver.many_calls == [] and resolver.single_calls == [] and recorder == []


def test_an_int_tool_id_reaches_the_resolver_as_a_string(recorder):
    seen = []

    class StrictResolver(RunResolver):
        def resolve_all(self, *, items):
            seen.extend(items)
            assert [i["toolkit_instance_id"] for i in items] == ["7"]
            assert all(type(i["toolkit_instance_id"]) is str for i in items)
            return super().resolve_all(items=items)

    sdk_tools.get_tools([_github_tool(tool_id=7)], SimpleNamespace(credential_resolver=StrictResolver()), llm=None)
    assert len(seen) == 1 and _built_tokens(recorder) == ["secret-7"]


def test_each_tools_scrub_set_is_its_own_when_the_third_toolkit_fails(monkeypatch):
    built = []

    def get_tools(tool):
        token = tool["settings"]["github_configuration"]["access_token"]
        if tool["id"] == 3:
            raise ValueError(f"echo {token} and earlier secret-1")
        built.append(token)
        return [_tool()]

    monkeypatch.setitem(sdk_tools.AVAILABLE_TOOLS, "github", {"get_tools": get_tools})
    with pytest.raises(ToolException) as exc:
        sdk_tools.get_tools(_six_tools(), SimpleNamespace(credential_resolver=RunResolver()), llm=None)
    assert built == ["secret-1", "secret-2"]
    assert "secret-3" not in str(exc.value) and cs.REDACTED in str(exc.value)
    assert "secret-1" in str(exc.value)    # only the failing toolkit's own released values are scrubbed


def test_the_runtime_build_clears_the_released_envelopes_after_the_loop(recorder, monkeypatch):
    captured = []
    real = runtime_tools.elitea_tools

    def spy(tools_list, elitea_client, llm, memory_store, **kwargs):
        captured.append(kwargs.get("released_slots"))
        return real(tools_list, elitea_client, llm, memory_store, **kwargs)

    monkeypatch.setattr(runtime_tools, "elitea_tools", spy)
    monkeypatch.setattr(runtime_tools, "community_tools", lambda *a, **k: [])
    monkeypatch.setattr(runtime_tools, "_mcp_tools", lambda *a, **k: [])
    tools_list = [_runtime_tool("github", n, github_settings()) for n in range(1, 4)]
    runtime_tools.get_tools(tools_list, elitea_client=SimpleNamespace(credential_resolver=RunResolver()))
    assert len(captured) == 3 and captured[0] is captured[1] is captured[2] and captured[0] == {}

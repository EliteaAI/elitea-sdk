"""Credential slots applied in the SDK at the last moment (C-1009 WP-1D slice 1, #6946).

A governed payload carries a slot, never the material. The SDK asks an injected resolver
for the value right before the toolkit is built and writes it, as plain `str`, into a local
copy only. A slot that cannot be resolved stops the run with a typed error; it is never
dropped, because e.g. the GitHub toolkit treats a missing token as anonymous access.
"""

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
    scrub,
)
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

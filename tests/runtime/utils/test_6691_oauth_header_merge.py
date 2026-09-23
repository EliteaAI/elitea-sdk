"""Pins merge_oauth_authorization's "configured PAT wins" precedence (#6691 Fault 2).

Re-breaks if the merge goes back to `headers.setdefault('Authorization', ...)`
(case-sensitive: a lower-case configured header would get a second, capitalized
key), if it starts overwriting a configured header unconditionally, or if it stops
treating an unresolved `{placeholder}` value as "not a credential".
"""

import pytest

from elitea_sdk.runtime.utils.mcp_oauth import (
    as_header_mapping,
    is_unresolved_mcp_placeholder,
    drop_unusable_authorization,
    has_authorization_on_the_wire,
    find_authorization_header,
    has_configured_authorization,
    is_unresolved_mcp_placeholder,
    merge_oauth_authorization,
)


def test_configured_pat_is_kept_over_an_oauth_token():
    headers = {"Authorization": "Bearer PAT-TOKEN"}

    merged, injected = merge_oauth_authorization(headers, "oauth-token")

    assert merged == {"Authorization": "Bearer PAT-TOKEN"}
    assert injected is False


def test_lower_case_configured_pat_is_kept_without_a_duplicate_key():
    headers = {"authorization": "Bearer PAT-TOKEN"}

    merged, injected = merge_oauth_authorization(headers, "oauth-token")

    assert merged == {"authorization": "Bearer PAT-TOKEN"}
    assert injected is False
    assert list(merged.keys()) == ["authorization"]


def test_absent_authorization_gets_the_oauth_token_injected():
    merged, injected = merge_oauth_authorization({"X-Foo": "1"}, "oauth-token")

    assert merged == {"X-Foo": "1", "Authorization": "Bearer oauth-token"}
    assert injected is True


@pytest.mark.parametrize("token", [None, ""])
def test_no_access_token_leaves_headers_unchanged(token):
    headers = {"X-Foo": "1"}

    merged, injected = merge_oauth_authorization(headers, token)

    assert merged == headers
    assert injected is False


def test_placeholder_authorization_is_replaced_by_the_oauth_token():
    headers = {"Authorization": "Bearer {github_token}"}

    merged, injected = merge_oauth_authorization(headers, "oauth-token")

    assert merged == {"Authorization": "Bearer oauth-token"}
    assert injected is True


@pytest.mark.parametrize("blank", ["", "   ", "Bearer ", "bearer", "Basic  ", "Token"])
def test_a_blank_credential_is_replaced_by_the_oauth_token(blank):
    merged, injected = merge_oauth_authorization({"Authorization": blank}, "oauth-token")

    assert merged == {"Authorization": "Bearer oauth-token"}
    assert injected is True


@pytest.mark.parametrize("value", ["", "Bearer ", "  token "])
def test_has_configured_authorization_ignores_a_blank_credential(value):
    assert has_configured_authorization({"Authorization": value}) is False


def test_a_bare_token_without_a_scheme_is_still_a_configured_credential():
    assert has_configured_authorization({"Authorization": "ghp_real"}) is True
    assert merge_oauth_authorization({"Authorization": "ghp_real"}, "oauth-token") == (
        {"Authorization": "ghp_real"}, False
    )


def test_merge_does_not_mutate_the_caller_headers_dict():
    configured = {"Authorization": "Bearer PAT-TOKEN"}
    merge_oauth_authorization(configured, "oauth-token")
    assert configured == {"Authorization": "Bearer PAT-TOKEN"}

    absent = {"X-Foo": "1"}
    merge_oauth_authorization(absent, "oauth-token")
    assert absent == {"X-Foo": "1"}

    placeholder = {"Authorization": "Bearer {github_token}"}
    merge_oauth_authorization(placeholder, "oauth-token")
    assert placeholder == {"Authorization": "Bearer {github_token}"}


def test_has_configured_authorization_ignores_an_unresolved_placeholder():
    assert has_configured_authorization({"Authorization": "Bearer {github_token}"}) is False
    assert has_configured_authorization({"Authorization": "Bearer real-token"}) is True
    assert has_configured_authorization({}) is False
    assert has_configured_authorization(None) is False


def test_find_authorization_header_is_case_insensitive():
    assert find_authorization_header({"authorization": "x"}) == "authorization"
    assert find_authorization_header({"Authorization": "x"}) == "Authorization"
    assert find_authorization_header({}) is None
    assert find_authorization_header(None) is None


@pytest.mark.parametrize(
    "value,expected",
    [
        ("{github_token}", True),
        ("Bearer {github_token}", True),
        ("real-token", False),
        ("", False),
        (123, False),
    ],
)
def test_is_unresolved_mcp_placeholder(value, expected):
    assert is_unresolved_mcp_placeholder(value) is expected


@pytest.mark.parametrize("value", ["Bearer {github_token}", "{token}", "Bearer ", "", "bearer"])
def test_a_value_that_is_not_a_credential_is_dropped_when_nothing_replaced_it(value):
    assert drop_unusable_authorization({"authorization": value, "X-A": "1"}) == {"X-A": "1"}


@pytest.mark.parametrize("value", ["Bearer pat", "ghp_abc", "Bearer x}{y"])
def test_a_real_credential_is_kept_on_the_wire(value):
    assert drop_unusable_authorization({"Authorization": value}) == {"Authorization": value}


def test_drop_unusable_authorization_returns_a_copy():
    headers = {"Authorization": "Bearer {x}"}
    assert drop_unusable_authorization(headers) == {} and headers == {"Authorization": "Bearer {x}"}


@pytest.mark.parametrize(
    ("headers", "injected", "expected"),
    [({"Authorization": "Bearer pat"}, False, True), ({"authorization": "Bearer "}, False, True),
     ({"Authorization": "Bearer t"}, True, False), ({}, False, False), (None, False, False)],
)
def test_whatever_authorization_is_sent_uninjected_counts_as_configured_for_the_401_message(headers, injected, expected):
    assert has_authorization_on_the_wire(headers, injected) is expected


@pytest.mark.parametrize("value", ["Bearer {github_token}", "{token}"])
def test_an_unfilled_prebuilt_template_is_not_a_credential(value):
    assert is_unresolved_mcp_placeholder(value) is True


@pytest.mark.parametrize("value", ["Bearer {{secret.github_pat}}", "{{ secret.pat }}"])
def test_an_unresolved_secret_reference_is_a_configured_credential_that_failed_to_resolve(value):
    assert is_unresolved_mcp_placeholder(value) is False
    assert has_configured_authorization({"Authorization": value}) is True
    assert merge_oauth_authorization({"Authorization": value}, "oauth-token") == ({"Authorization": value}, False)
    assert drop_unusable_authorization({"Authorization": value}) == {"Authorization": value}


@pytest.mark.parametrize("value", ["Bearer x}{y", "Bearer {a b}", "Bearer {}", "Bearer {{not.a.secret}}", "Bearer {{}}", "Bearer {a-b}"])
def test_a_credential_that_merely_contains_braces_stays_a_credential(value):
    assert is_unresolved_mcp_placeholder(value) is False
    assert has_configured_authorization({"Authorization": value}) is True
    merged, injected = merge_oauth_authorization({"Authorization": value}, "oauth-token")
    assert (merged, injected) == ({"Authorization": value}, False)


def test_headers_stored_as_a_json_string_are_parsed_for_the_credential_helpers():
    assert as_header_mapping('{"Authorization": "Bearer pat"}') == {"Authorization": "Bearer pat"}


@pytest.mark.parametrize("value", ["not json at all", '["a", "b"]', "", None, {"X-A": "1"}])
def test_anything_else_is_handed_on_untouched(value):
    assert as_header_mapping(value) == value


def test_the_helpers_hand_a_non_mapping_on_untouched_for_the_toolkit_to_report():
    assert merge_oauth_authorization("not json at all", "oauth-token") == ("not json at all", False)
    assert drop_unusable_authorization("not json at all") == "not json at all"
    assert has_configured_authorization("not json at all") is False
    assert has_authorization_on_the_wire("not json at all", False) is False

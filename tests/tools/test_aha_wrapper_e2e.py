"""End-to-end Aha! toolkit tests — hits a live Aha! account.

These tests are opt-in and only run when ``AHA_BASE_URL`` and ``AHA_API_KEY``
are set (either in the process environment or in a ``.env`` file loaded
via ``python-dotenv``).

.env keys read by this suite
----------------------------
Required:
  AHA_BASE_URL    e.g. https://mycompany.aha.io
  AHA_API_KEY     Aha! personal or project API token

Optional (unlock more targeted tests):
  AHA_PRODUCT_ID       product reference/ID, e.g. DEVELOP
  AHA_FEATURE_REF      feature reference number, e.g. DEVELOP-123
  AHA_REQUIREMENT_REF  requirement reference number, e.g. DEVELOP-123-1
  AHA_RELEASE_REF      release reference number, e.g. DEVELOP-R-1
  AHA_PAGE_REF         page reference number, e.g. ABC-N-1
  AHA_SEARCH_QUERY     free-text query used by search / search_documents (default: "test")

Skipping
--------
Tests that need optional refs are individually skipped when the ref is
absent, so the suite is safe to run with only the required credentials.

Running
-------
    pytest tests/tools/test_aha_wrapper_e2e.py -v
"""

from __future__ import annotations

import os

import pytest
from langchain_core.tools import ToolException
from pydantic import SecretStr

try:
    from dotenv import load_dotenv

    load_dotenv()
    # Also honour Elitea's convention of .elitea/.env / .alita/.env if present.
    for candidate in (".elitea/.env", ".alita/.env"):
        if os.path.exists(candidate):
            load_dotenv(candidate, override=False)
except ImportError:  # dotenv is optional; skip silently if missing
    pass

from elitea_sdk.tools.aha.api_wrapper import AhaApiWrapper

BASE_URL = os.getenv("AHA_BASE_URL")
API_KEY = os.getenv("AHA_API_KEY")

# Module-level skip: without credentials there's nothing to test.
pytestmark = pytest.mark.skipif(
    not BASE_URL or not API_KEY,
    reason="AHA_BASE_URL and AHA_API_KEY must be set to run Aha! e2e tests",
)


@pytest.fixture(scope="module")
def wrapper() -> AhaApiWrapper:
    return AhaApiWrapper(base_url=BASE_URL, api_key=SecretStr(API_KEY))


def _skip_without(env_var: str) -> str:
    value = os.getenv(env_var)
    if not value:
        pytest.skip(f"{env_var} not set — skipping test that needs it")
    return value


def _skip_on_permission_denied(exc: ToolException) -> None:
    """If Aha! rejects the call with 403, skip rather than fail.

    Some list endpoints require account-admin permissions even when
    scoped by product (Aha! restricts them for the 'product owner' role
    and below).
    """
    if "403" in str(exc) or "Access denied" in str(exc):
        pytest.skip(f"Aha! returned 403 (insufficient role): {exc}")


# -----------------------------------------------------------------------------
# Connection / auth sanity
# -----------------------------------------------------------------------------


def test_credentials_accepted_by_listing_products(wrapper):
    """Smoke-test auth by fetching a single page of products.

    Aha returns 401 on bad tokens, which surfaces as ToolException. A clean
    call with any non-empty list (or empty list without error) means the
    token authenticated successfully.
    """
    result = wrapper.list_products(per_page=1, max_records=1)
    assert isinstance(result, list)


def test_find_project_dispatcher_works(wrapper):
    result = wrapper.find_project(per_page=1, max_records=1)
    assert isinstance(result, list)


# -----------------------------------------------------------------------------
# REST reads (opt-in via env-provided references)
# -----------------------------------------------------------------------------


def test_get_product(wrapper):
    product_ref = _skip_without("AHA_PRODUCT_ID")
    record = wrapper.get_product(product_ref)
    assert isinstance(record, dict)
    assert record  # non-empty


def test_get_feature(wrapper):
    feature_ref = _skip_without("AHA_FEATURE_REF")
    record = wrapper.get_feature(feature_ref)
    assert isinstance(record, dict)
    # Sanity check: reference_num should match what we asked for
    if "reference_num" in record:
        assert record["reference_num"] == feature_ref


def test_get_feature_projection(wrapper):
    feature_ref = _skip_without("AHA_FEATURE_REF")
    record = wrapper.get_feature(feature_ref, fields=["id", "reference_num", "name"])
    assert set(record.keys()) <= {"id", "reference_num", "name"}


def test_get_requirement(wrapper):
    req_ref = _skip_without("AHA_REQUIREMENT_REF")
    record = wrapper.get_requirement(req_ref)
    assert isinstance(record, dict)


def test_get_release(wrapper):
    release_ref = _skip_without("AHA_RELEASE_REF")
    record = wrapper.get_release(release_ref)
    assert isinstance(record, dict)


# -----------------------------------------------------------------------------
# REST lists
# -----------------------------------------------------------------------------


def test_list_features_scoped_to_product(wrapper):
    # Aha! restricts un-scoped /features to account-level admins; scope by
    # product to stay within the permissions of a typical API token.
    # Even scoped, the 'product owner' role can still be denied — skip on 403.
    product_ref = _skip_without("AHA_PRODUCT_ID")
    try:
        result = wrapper.list_features(product_id=product_ref, per_page=5, max_records=5)
    except ToolException as exc:
        _skip_on_permission_denied(exc)
        raise
    assert isinstance(result, list)


def test_list_releases_scoped_to_product(wrapper):
    # Same story as list_features: un-scoped /releases requires elevated
    # permissions; scope the call via product for a portable test.
    product_ref = _skip_without("AHA_PRODUCT_ID")
    try:
        result = wrapper.list_releases(product_id=product_ref, per_page=5, max_records=5)
    except ToolException as exc:
        _skip_on_permission_denied(exc)
        raise
    assert isinstance(result, list)


# -----------------------------------------------------------------------------
# Search
# -----------------------------------------------------------------------------


def test_search(wrapper):
    query = os.getenv("AHA_SEARCH_QUERY", "test")
    result = wrapper.search(q=query, per_page=5, max_records=5)
    assert isinstance(result, list)


def test_search_records_dispatcher(wrapper):
    # Dispatcher routes 'feature' → list_features; scope by product to avoid
    # the un-scoped /features endpoint that requires elevated permissions.
    product_ref = _skip_without("AHA_PRODUCT_ID")
    query = os.getenv("AHA_SEARCH_QUERY", "test")
    try:
        result = wrapper.search_records(
            record_type="feature",
            product_id=product_ref,
            q=query,
            per_page=5,
            max_records=5,
        )
    except ToolException as exc:
        _skip_on_permission_denied(exc)
        raise
    assert isinstance(result, list)


# -----------------------------------------------------------------------------
# GraphQL reads
# -----------------------------------------------------------------------------


def test_get_page_via_graphql(wrapper):
    page_ref = _skip_without("AHA_PAGE_REF")
    record = wrapper.get_page(page_ref)
    assert isinstance(record, dict)


def test_search_documents(wrapper):
    query = os.getenv("AHA_SEARCH_QUERY", "test")
    nodes = wrapper.search_documents(query)
    assert isinstance(nodes, list)


def test_get_feature_gql(wrapper):
    feature_ref = _skip_without("AHA_FEATURE_REF")
    record = wrapper.get_feature_gql(feature_ref)
    assert isinstance(record, dict)


# -----------------------------------------------------------------------------
# Metadata endpoints
# -----------------------------------------------------------------------------


def test_fields_metadata(wrapper):
    # Aha! REST v1 no longer exposes /custom_fields on all accounts (returns
    # 404 on modern tenants). Treat that specific failure as an environment
    # limitation rather than a code defect.
    try:
        result = wrapper.fields_metadata(per_page=5, max_records=5)
    except ToolException as exc:
        if "404" in str(exc):
            pytest.skip(
                "Aha! /custom_fields endpoint not available on this account "
                "(HTTP 404). Skipping metadata check."
            )
        raise
    assert isinstance(result, list)


# -----------------------------------------------------------------------------
# Read-side dispatcher
# -----------------------------------------------------------------------------


def test_read_records_feature(wrapper):
    feature_ref = _skip_without("AHA_FEATURE_REF")
    record = wrapper.read_records(record_type="feature", reference_or_id=feature_ref)
    assert isinstance(record, dict)
    assert record


def test_read_records_page(wrapper):
    page_ref = _skip_without("AHA_PAGE_REF")
    record = wrapper.read_records(record_type="page", reference_or_id=page_ref)
    assert isinstance(record, dict)


# -----------------------------------------------------------------------------
# Configuration-level connection check (mirrors platform behaviour)
# -----------------------------------------------------------------------------


def test_configuration_check_connection_succeeds():
    """AhaConfiguration.check_connection is what the platform runs on save."""
    from elitea_sdk.configurations.aha import AhaConfiguration

    err = AhaConfiguration.check_connection({"base_url": BASE_URL, "api_key": SecretStr(API_KEY)})
    assert err is None, f"Aha connection check failed: {err}"


def test_configuration_check_connection_rejects_bad_token():
    from elitea_sdk.configurations.aha import AhaConfiguration

    err = AhaConfiguration.check_connection(
        {"base_url": BASE_URL, "api_key": SecretStr("definitely-not-a-real-token")}
    )
    assert err is not None
    # Should be a friendly auth message, not a raw traceback / secret leak.
    assert "not-a-real-token" not in err

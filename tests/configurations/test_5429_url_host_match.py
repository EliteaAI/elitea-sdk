"""Regression tests for code-scanning rule py/incomplete-url-substring-sanitization
(issue #5429).

Six sites used `domain in url` substring checks to decide Atlassian Cloud vs
Server and Bitbucket Cloud vs Server. Substring matching is spoofable
(e.g. https://evil.atlassian.net.attacker.com). They now route through
url_host_matches_domain, which matches on the parsed host.

These tests assert:
1. url_host_matches_domain accepts legit hosts and rejects spoofed/malformed ones.
2. The callers (_hosting_to_cloud, _resolve_api_version,
   _resolve_confluence_api_version) keep identical behavior for legit inputs
   and now resolve spoofed URLs to the safe Server/non-cloud branch.
"""

import pytest

from elitea_sdk.configurations.utils import (
    url_host_matches_domain,
    _hosting_to_cloud,
    _resolve_api_version,
    _resolve_confluence_api_version,
)


class TestUrlHostMatchesDomain:
    @pytest.mark.parametrize("url", [
        "https://company.atlassian.net",
        "https://company.atlassian.net/wiki",
        "http://company.atlassian.net",
        "https://deep.sub.atlassian.net/x",
        "atlassian.net",                       # bare apex, no scheme
        "company.atlassian.net/foo",           # subdomain, no scheme
        "HTTPS://Company.Atlassian.NET",       # case-insensitive
    ])
    def test_accepts_legit_atlassian(self, url):
        assert url_host_matches_domain(url, "atlassian.net") is True

    @pytest.mark.parametrize("url", [
        "https://evil.atlassian.net.attacker.com",   # suffix spoof
        "https://attacker.com/?x=.atlassian.net",    # query spoof
        "https://atlassian.net.evil.com",            # apex-as-subdomain spoof
        "https://notatlassian.net",                  # missing dot boundary
        "https://company.example.com",               # unrelated
        "",                                          # empty
        None,                                        # None
    ])
    def test_rejects_spoofed_or_empty(self, url):
        assert url_host_matches_domain(url, "atlassian.net") is False

    @pytest.mark.parametrize("url,expected", [
        ("https://bitbucket.org/team/repo", True),
        ("https://api.bitbucket.org/2.0/user", True),   # subdomain covered
        ("bitbucket.org", True),
        ("https://bitbucket.org.evil.com", False),      # spoof
        ("https://mycompany-bitbucket.com", False),     # 'bitbucket' substring, wrong host
        ("https://bitbucket.mycompany.com", False),     # self-hosted server
    ])
    def test_bitbucket_domain(self, url, expected):
        assert url_host_matches_domain(url, "bitbucket.org") is expected


class TestNoRegressionInCallers:
    # _hosting_to_cloud: auto-detect path
    def test_hosting_auto_cloud_url(self):
        assert _hosting_to_cloud("auto", "https://x.atlassian.net") is True

    def test_hosting_auto_server_url(self):
        assert _hosting_to_cloud("auto", "https://jira.mycorp.com") is False

    def test_hosting_explicit_unchanged(self):
        assert _hosting_to_cloud("cloud", "https://jira.mycorp.com") is True
        assert _hosting_to_cloud("server", "https://x.atlassian.net") is False

    def test_hosting_auto_spoof_now_server(self):
        # Previously the substring check returned True (Cloud) for this spoof.
        assert _hosting_to_cloud("auto", "https://evil.atlassian.net.attacker.com") is False

    # _resolve_api_version (Jira): auto path
    def test_jira_api_explicit_kept(self):
        assert _resolve_api_version("2", None, "https://x.atlassian.net") == "2"
        assert _resolve_api_version("3", None, "https://jira.mycorp.com") == "3"

    def test_jira_api_auto_cloud(self):
        assert _resolve_api_version("auto", None, "https://x.atlassian.net") == "3"

    def test_jira_api_auto_server(self):
        assert _resolve_api_version("auto", None, "https://jira.mycorp.com") == "2"

    def test_jira_api_auto_spoof_now_server(self):
        assert _resolve_api_version("auto", None, "https://x.atlassian.net.evil.com") == "2"

    # _resolve_confluence_api_version: auto path
    def test_confluence_api_auto_cloud(self):
        assert _resolve_confluence_api_version("auto", None, "https://x.atlassian.net") == "2"

    def test_confluence_api_auto_server(self):
        assert _resolve_confluence_api_version("auto", None, "https://conf.mycorp.com") == "1"

    def test_confluence_api_auto_spoof_now_server(self):
        assert _resolve_confluence_api_version("auto", None, "https://x.atlassian.net.evil.com") == "1"

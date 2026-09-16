# Copyright (c) 2026 EPAM Systems
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pins the GitHub client's request pacing (#6592)."""

import ast
import inspect

from github import Github, GithubIntegration

from elitea_sdk.tools.github import github_client
from elitea_sdk.tools.github.github_client import (
    GITHUB_DOCUMENTED_GET_LIMIT_PER_SECOND,
    GITHUB_SECONDS_BETWEEN_REQUESTS,
)

SHARE_OF_LIMIT_THE_INDEXER_MAY_TAKE = 0.7
PACED_CLIENT_CLASSES = ("Github", "GithubIntegration")
PACING_KEYWORD = "seconds_between_requests"
PACING_CONSTANT = "GITHUB_SECONDS_BETWEEN_REQUESTS"
CLIENT_CONSTRUCTION_PATHS = 3


def pygithub_default_pacing():
    return inspect.signature(Github.__init__).parameters[PACING_KEYWORD].default


def configured_requests_per_second():
    return 1 / GITHUB_SECONDS_BETWEEN_REQUESTS


def client_constructions():
    tree = ast.parse(inspect.getsource(github_client))
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in PACED_CLIENT_CLASSES
    ]


def pacing_argument_of(node):
    for keyword in node.keywords:
        if keyword.arg == PACING_KEYWORD:
            return keyword.value
    return None


class TestTheConfiguredPacing:
    def test_it_is_faster_than_the_pygithub_default(self):
        assert GITHUB_SECONDS_BETWEEN_REQUESTS < pygithub_default_pacing()

    def test_it_stays_under_the_documented_secondary_limit(self):
        assert configured_requests_per_second() <= GITHUB_DOCUMENTED_GET_LIMIT_PER_SECOND

    def test_it_leaves_headroom_for_everything_else_sharing_the_token(self):
        assert configured_requests_per_second() <= (
            GITHUB_DOCUMENTED_GET_LIMIT_PER_SECOND * SHARE_OF_LIMIT_THE_INDEXER_MAY_TAKE
        )

    def test_both_client_classes_accept_the_setting(self):
        for cls in (Github, GithubIntegration):
            assert PACING_KEYWORD in inspect.signature(cls.__init__).parameters


class TestEveryClientPathIsPaced:
    def test_every_construction_path_is_still_covered(self):
        assert len(client_constructions()) == CLIENT_CONSTRUCTION_PATHS

    def test_no_client_is_constructed_on_the_pygithub_default(self):
        unpaced = [
            f"{node.func.id} at line {node.lineno}"
            for node in client_constructions()
            if pacing_argument_of(node) is None
        ]
        assert not unpaced, f"client(s) left on the PyGithub default: {unpaced}"

    def test_no_path_inlines_its_own_pacing_literal(self):
        drifted = [
            f"{node.func.id} at line {node.lineno}"
            for node in client_constructions()
            if not isinstance(pacing_argument_of(node), ast.Name)
            or pacing_argument_of(node).id != PACING_CONSTANT
        ]
        assert not drifted, f"client(s) not using {PACING_CONSTANT}: {drifted}"

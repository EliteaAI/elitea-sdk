"""Tests for GitHub get_issue returning the comment thread (#4957).

Covers:
  * Comments come back with body, author and ISO timestamp.
  * An issue with no comments returns an empty list, not an error.
  * A thread longer than the limit returns the FIRST N in order, plus the true
    total and a note, so the agent knows it is seeing a partial thread.
  * The result stays JSON-serializable (no raw datetime leaks).
  * A comment whose author was deleted reports user=None rather than raising.
  * A failure reading comments still returns the issue itself.
  * Cost is bounded at one or two page fetches, however deep the window sits.
  * The limit is exactly one GitHub page, which is what makes that pass one request.
"""

import inspect
import json
import re

from elitea_sdk.runtime.tool_result_bounds import bound_tool_result
from datetime import datetime

from github import Github

import pytest

from elitea_sdk.runtime.utils import trace_limits
from elitea_sdk.runtime.utils.trace_limits import estimate_chars, resolve_tool_result_limit
from elitea_sdk.tools.github import name as GITHUB_TOOLKIT_NAME
from elitea_sdk.tools.github.github_client import (
    GITHUB_DEFAULT_PAGE_SIZE,
    GITHUB_ISSUE_COMMENTS_LIMIT,
    GITHUB_ISSUE_COMMENTS_NOTE_RESERVE,
    GITHUB_TOOLKIT_TYPE,
    GitHubClient,
)


@pytest.fixture
def bounding_limits():
    """Restore the process-wide bounds; configure_tool_result_limits sets module globals."""
    saved = (trace_limits._bounding_enabled, trace_limits._bounding_limit,
             dict(trace_limits._bounding_per_toolkit))
    yield trace_limits.configure_tool_result_limits
    (trace_limits._bounding_enabled, trace_limits._bounding_limit,
     trace_limits._bounding_per_toolkit) = saved


class FakeUser:
    def __init__(self, login):
        self.login = login


class FakeComment:
    def __init__(self, body, user=FakeUser('alice'), created_at=None, url=None):
        self.body = body
        self.user = user
        self.created_at = created_at or datetime(2026, 1, 3, 10, 5, 0)
        self.html_url = url or 'https://github.com/owner/repo/issues/42#issuecomment-1'


class FakeLabel:
    def __init__(self, name):
        self.name = name


class FakePaginatedComments:
    """PyGithub's PaginatedList semantics: get_page is 0-indexed and costs one HTTP fetch."""

    def __init__(self, comments, page_size=GITHUB_DEFAULT_PAGE_SIZE):
        self._comments = comments
        self._page_size = page_size
        self.page_fetches = 0

    def get_page(self, index):
        self.page_fetches += 1
        start = index * self._page_size
        return self._comments[start:start + self._page_size]

    def __iter__(self):
        index = 0
        while index < len(self._comments):
            self.page_fetches += 1
            yield from self._comments[index:index + self._page_size]
            index += self._page_size


class FakeIssue:
    def __init__(self, comments, comments_total=None, raises=None):
        self.number = 42
        self.title = 'Login is broken'
        self.body = 'Steps to reproduce'
        self.state = 'open'
        self.html_url = 'https://github.com/owner/repo/issues/42'
        self.created_at = datetime(2026, 1, 1, 9, 0, 0)
        self.updated_at = datetime(2026, 1, 2, 9, 0, 0)
        self.labels = [FakeLabel('bug')]
        self.assignees = [FakeUser('bob')]
        self.comments = comments_total if comments_total is not None else len(comments)
        self._comments = comments
        self._raises = raises
        self.get_comments_calls = 0

    def get_comments(self):
        self.get_comments_calls += 1
        if self._raises:
            raise self._raises
        self.paginated = FakePaginatedComments(self._comments)
        return self.paginated


class FakeRepo:
    def __init__(self, issue):
        self.issue = issue

    def get_issue(self, number):
        return self.issue


class FakeGitHubApi:
    def __init__(self, repo):
        self.repo = repo

    def get_repo(self, repo_name):
        return self.repo


def _make_client(issue: FakeIssue) -> GitHubClient:
    return GitHubClient.model_construct(
        github_repository='owner/repo',
        active_branch='main',
        github_base_branch='main',
        github_api=FakeGitHubApi(FakeRepo(issue)),
        elitea=None,
    )


def test_comments_include_body_author_and_timestamp():
    issue = FakeIssue([FakeComment('Looks like a regression')])
    result = _make_client(issue).get_issue(42)

    assert result['comments'] == [
        {
            'body': 'Looks like a regression',
            'user': 'alice',
            'created_at': '2026-01-03T10:05:00',
            'url': 'https://github.com/owner/repo/issues/42#issuecomment-1',
        }
    ]
    assert result['comments_total'] == 1
    assert 'comments_note' not in result


def test_issue_without_comments_returns_empty_list():
    issue = FakeIssue([])
    result = _make_client(issue).get_issue(42)

    assert isinstance(result, dict)
    assert result['comments'] == []
    assert result['comments_total'] == 0
    assert 'comments_note' not in result
    assert result['title'] == 'Login is broken'


def test_over_limit_returns_the_first_n_and_the_true_total():
    total = GITHUB_ISSUE_COMMENTS_LIMIT + 10
    issue = FakeIssue([FakeComment(f'comment {i}') for i in range(total)])
    result = _make_client(issue).get_issue(42)

    assert len(result['comments']) == GITHUB_ISSUE_COMMENTS_LIMIT
    assert [c['body'] for c in result['comments']] == [
        f'comment {i}' for i in range(GITHUB_ISSUE_COMMENTS_LIMIT)
    ]
    assert result['comments_total'] == total
    assert str(total) in result['comments_note']
    assert str(GITHUB_ISSUE_COMMENTS_LIMIT) in result['comments_note']


def test_note_names_an_offset_that_actually_reaches_the_rest():
    """The note must be actionable: following it verbatim must return the tail."""
    total = GITHUB_ISSUE_COMMENTS_LIMIT + 10
    bodies = [f'comment {i}' for i in range(total)]
    client = _make_client(FakeIssue([FakeComment(b) for b in bodies]))

    first = client.get_issue(42)
    offset = int(re.search(r'comments_offset=(\d+)', first['comments_note']).group(1))

    rest = _make_client(FakeIssue([FakeComment(b) for b in bodies])).get_issue(
        42, comments_offset=offset
    )

    assert [c['body'] for c in rest['comments']] == bodies[offset:]
    assert 'comments_note' not in rest
    seen = [c['body'] for c in first['comments']] + [c['body'] for c in rest['comments']]
    assert seen == bodies


def test_offset_window_is_bounded_by_the_limit():
    total = GITHUB_ISSUE_COMMENTS_LIMIT * 3
    bodies = [f'comment {i}' for i in range(total)]
    issue = FakeIssue([FakeComment(b) for b in bodies])

    result = _make_client(issue).get_issue(42, comments_offset=GITHUB_ISSUE_COMMENTS_LIMIT)

    assert [c['body'] for c in result['comments']] == bodies[
        GITHUB_ISSUE_COMMENTS_LIMIT:GITHUB_ISSUE_COMMENTS_LIMIT * 2
    ]
    assert f'comments_offset={GITHUB_ISSUE_COMMENTS_LIMIT * 2}' in result['comments_note']


def test_offset_past_the_end_says_so_rather_than_looking_like_an_empty_thread():
    """The description tells the model that no comments_note means it has the whole thread,
    so a silent empty window reads as "this issue has no comments"."""
    issue = FakeIssue([FakeComment('only one')])

    result = _make_client(issue).get_issue(42, comments_offset=500)

    assert result['comments'] == []
    assert result['comments_total'] == 1
    assert 'the issue reports 1' in result['comments_note']
    assert '500' in result['comments_note']
    assert re.findall(r'comments_offset=(\d+)', result['comments_note']) == [], (
        'on a thread shorter than one window the only offset to suggest is 0, which is '
        'where a caller walking the thread already came from -- that closes a cycle'
    )
    assert result['title'] == 'Login is broken'


def test_offset_past_the_end_of_an_empty_thread_stays_silent():
    """With no comments at all there is nothing to point at, and a note would contradict
    comments_total == 0."""
    issue = FakeIssue([])

    result = _make_client(issue).get_issue(42, comments_offset=500)

    assert result['comments'] == []
    assert result['comments_total'] == 0
    assert 'comments_note' not in result


def test_negative_offset_is_floored_to_the_start_of_the_thread():
    """divmod(-5, 30) is (-1, 25), which would ask PyGithub for page -1."""
    bodies = [f'comment {i}' for i in range(5)]
    issue = FakeIssue([FakeComment(b) for b in bodies])

    result = _make_client(issue).get_issue(42, comments_offset=-5)

    assert [c['body'] for c in result['comments']] == bodies
    assert 'comments_note' not in result


def test_unparseable_offset_names_the_parameter_rather_than_blaming_the_fetch():
    """The model can only correct an offset it is told is the problem; "Failed to get issue"
    reads as the issue being unreachable."""
    issue = FakeIssue([FakeComment('only one')])

    result = _make_client(issue).get_issue(42, comments_offset='abc')

    assert 'comments_offset' in result
    assert "'abc'" in result
    assert 'Failed to get issue' not in result


def test_offset_given_as_a_string_is_honoured():
    bodies = [f'comment {i}' for i in range(60)]
    issue = FakeIssue([FakeComment(b) for b in bodies])

    result = _make_client(issue).get_issue(42, comments_offset='30')

    assert [c['body'] for c in result['comments']] == bodies[30:60]


def test_comments_offset_is_declared_on_the_args_schema():
    """#6532: a tool may not read a parameter its args_schema never advertises."""
    client = _make_client(FakeIssue([]))
    schema = next(
        tool['args_schema']
        for tool in client.get_available_tools()
        if tool['name'] == 'get_issue'
    )

    assert 'comments_offset' in schema.model_fields


def test_result_is_json_serializable():
    issue = FakeIssue([FakeComment('Looks good')])
    result = _make_client(issue).get_issue(42)

    assert json.loads(json.dumps(result)) == result
    assert isinstance(result['comments'][0]['created_at'], str)


def test_deleted_comment_author_is_reported_as_none():
    issue = FakeIssue([FakeComment('Posted by a since-deleted account', user=None)])
    result = _make_client(issue).get_issue(42)

    assert result['comments'][0]['user'] is None
    assert result['comments'][0]['body'] == 'Posted by a since-deleted account'


def test_comment_failure_does_not_lose_the_issue():
    issue = FakeIssue([], raises=RuntimeError('403 Forbidden'))
    result = _make_client(issue).get_issue(42)

    assert result['title'] == 'Login is broken'
    assert result['comments'] == []
    assert '403 Forbidden' in result['comments_error']


def test_long_thread_is_not_walked_beyond_the_limit():
    """A 400-comment thread must cost ONE page fetch, not ceil(400/30)=14."""
    issue = FakeIssue([FakeComment(f'comment {i}') for i in range(400)])
    _make_client(issue).get_issue(42)

    assert issue.get_comments_calls == 1
    assert issue.paginated.page_fetches == 1


def test_deep_offset_costs_one_page_fetch_not_one_per_page_skipped():
    """The cost must not grow with the offset: offset=360 indexes to page 12 directly."""
    issue = FakeIssue([FakeComment(f'comment {i}') for i in range(400)])

    result = _make_client(issue).get_issue(42, comments_offset=360)

    assert [c['body'] for c in result['comments']] == [f'comment {i}' for i in range(360, 390)]
    assert issue.paginated.page_fetches == 1


def test_unaligned_offset_spans_at_most_two_page_fetches():
    issue = FakeIssue([FakeComment(f'comment {i}') for i in range(400)])

    result = _make_client(issue).get_issue(42, comments_offset=365)

    assert [c['body'] for c in result['comments']] == [f'comment {i}' for i in range(365, 395)]
    assert issue.paginated.page_fetches == 2


def test_comment_limit_is_exactly_one_github_page():
    assert GITHUB_ISSUE_COMMENTS_LIMIT == GITHUB_DEFAULT_PAGE_SIZE == 30


def test_page_size_constant_matches_the_pygithub_default():
    assert inspect.signature(Github.__init__).parameters['per_page'].default == GITHUB_DEFAULT_PAGE_SIZE


def test_registered_description_states_the_actual_limit():
    client = _make_client(FakeIssue([]))
    description = next(
        tool['description']
        for tool in client.get_available_tools()
        if tool['name'] == 'get_issue'
    )

    assert str(GITHUB_ISSUE_COMMENTS_LIMIT) in description
    assert '{comments_limit}' not in description
    assert 'comments_total' in description
    assert 'comments_note' in description


class TestNoteAlwaysMakesProgress:
    """A note that names the offset just requested makes the model loop until it
    exhausts its step budget -- the description tells it to repeat until the note is
    gone. Every note must therefore advance past what was asked for."""

    def test_comment_read_failure_emits_no_note(self):
        issue = FakeIssue([], comments_total=57, raises=RuntimeError('502 Bad Gateway'))

        result = _make_client(issue).get_issue(42)

        assert result['comments'] == []
        assert '502 Bad Gateway' in result['comments_error']
        assert 'comments_note' not in result

    def test_comments_total_overstating_the_thread_terminates(self):
        """GitHub's count can exceed what get_comments returns (deletion race)."""
        bodies = [f'comment {i}' for i in range(50)]

        offsets_seen = []
        offset = 0
        for _ in range(6):
            issue = FakeIssue([FakeComment(b) for b in bodies], comments_total=57)
            result = _make_client(issue).get_issue(42, comments_offset=offset)
            offsets_seen.append(offset)
            nxt = re.findall(r'comments_offset=(\d+)', result.get('comments_note') or '')
            if not nxt:
                break
            offset = int(nxt[0])
            assert offset not in offsets_seen, f'note repeated offset {offset}'
        else:
            raise AssertionError('notes never stopped')

        assert offsets_seen == [0, 30, 50]


class TestWindowIsSelfLimiting:
    """comments_note names a resume offset, so the window must be bounded HERE. If the
    runtime bounder shrinks the list afterwards it drops whole comments, and the note
    would then resume past comments the caller never saw."""

    def test_huge_comments_shrink_the_window_and_the_resume_offset_follows(self):
        big = 'x' * 50_000
        issue = FakeIssue([FakeComment(big) for _ in range(GITHUB_ISSUE_COMMENTS_LIMIT)],
                          comments_total=137)

        result = _make_client(issue).get_issue(42)

        assert len(result['comments']) < GITHUB_ISSUE_COMMENTS_LIMIT
        assert f"comments_offset={len(result['comments'])}" in result['comments_note']
        assert f"1-{len(result['comments'])} of 137" in result['comments_note']

    def test_whole_payload_fits_the_limit_so_the_issue_body_survives(self):
        """The budget must cover the envelope, not just comment bodies: otherwise the
        payload lands over the limit and the bounder charges it to the largest leaf,
        which is the issue description the caller asked for."""
        issue = FakeIssue([FakeComment('x' * 6_600) for _ in range(GITHUB_ISSUE_COMMENTS_LIMIT)],
                          comments_total=137)
        issue.body = 'd' * 10_400

        result = _make_client(issue).get_issue(42)

        assert estimate_chars(result) <= resolve_tool_result_limit(GITHUB_TOOLKIT_TYPE)
        bounded, original = bound_tool_result(result, 'get_issue', GITHUB_TOOLKIT_TYPE)
        assert original is None
        assert len(bounded['body']) == 10_400

    def test_payload_never_exceeds_the_limit_across_the_boundary_band(self):
        """Swept in BOTH dimensions rather than spot-checked. The per-comment envelope is
        only ~113 chars, so counting bodies alone overshoots ONLY where a full window lands
        within that margin of the limit -- 6,600 admits 30 and overshoots while 6,650 admits
        29 and fits, so one comment size cannot pin it. The issue body moves that boundary
        too (it is subtracted from the budget), so each body size has its own band: ~6,650
        at an empty description, ~4,450 at GitHub's 65,536-char maximum."""
        limit = resolve_tool_result_limit(GITHUB_TOOLKIT_TYPE)
        overshoots = {}
        slack = limit
        for body_len in (0, 1_000, 20_000, 65_536):
            for comment_len in range(4_000, 7_001, 100):
                issue = FakeIssue([FakeComment('x' * comment_len)
                                   for _ in range(GITHUB_ISSUE_COMMENTS_LIMIT)], comments_total=137)
                issue.body = 'd' * body_len
                measured = estimate_chars(_make_client(issue).get_issue(42))
                if measured > limit:
                    overshoots[(body_len, comment_len)] = measured - limit
                slack = min(slack, limit - measured)

        assert not overshoots, f'payload exceeded the limit at (body, comment): {overshoots}'
        assert slack >= 250, (
            f'only {slack} chars of slack left: GITHUB_ISSUE_COMMENTS_NOTE_RESERVE covers the '
            f'note and list overhead added after the budget is measured, and nothing else pins it'
        )

    def test_a_per_toolkit_limit_override_shrinks_the_window(self, bounding_limits):
        """The enforced limit is resolve_tool_result_limit(toolkit), which an admin can
        lower per toolkit; a constant bound at import would overfill it."""
        bounding_limits(enabled=True, limit=200_000, per_toolkit={GITHUB_TOOLKIT_TYPE: 20_000})
        issue = FakeIssue([FakeComment('x' * 6_000) for _ in range(GITHUB_ISSUE_COMMENTS_LIMIT)],
                          comments_total=137)

        result = _make_client(issue).get_issue(42)

        assert len(result['comments']) < GITHUB_ISSUE_COMMENTS_LIMIT
        assert estimate_chars(result) <= 20_000
        assert f"comments_offset={len(result['comments'])}" in result['comments_note']

    def test_one_oversized_comment_is_still_returned(self):
        """Returning nothing would make the note unable to advance, re-creating the loop."""
        issue = FakeIssue([FakeComment('x' * 300_000), FakeComment('second')], comments_total=2)

        result = _make_client(issue).get_issue(42)

        assert len(result['comments']) == 1
        assert 'comments_offset=1' in result['comments_note']

    def test_a_lone_over_budget_comment_is_cut_to_fit_and_says_so(self, bounding_limits):
        """A budget too small for even one comment must not hand the bounder a payload it
        would answer by dropping that comment, leaving the note promising what was lost."""
        bounding_limits(enabled=True, limit=200_000, per_toolkit={GITHUB_TOOLKIT_TYPE: 20_000})
        issue = FakeIssue([FakeComment('x' * 60_000), FakeComment('second')], comments_total=2)
        issue.body = 'd' * 5_000

        result = _make_client(issue).get_issue(42)

        assert len(result['comments']) == 1
        assert '[truncated: 60000 chars' in result['comments'][0]['body']
        assert len(result['comments'][0]['body']) < 60_000
        assert estimate_chars(result) <= 20_000
        bounded, original = bound_tool_result(result, 'get_issue', GITHUB_TOOLKIT_TYPE)
        assert original is None
        assert 'comments_offset=1' in result['comments_note']

    def test_when_nothing_fits_the_description_is_kept_and_the_note_advances(self, bounding_limits):
        """A description that nearly fills the limit leaves no room for even one comment's
        metadata. Carrying a content-free comment anyway costs description characters for
        nothing, so the comment is withheld, named by url, and the offset advances past it."""
        bounding_limits(enabled=True, limit=200_000, per_toolkit={GITHUB_TOOLKIT_TYPE: 20_000})
        issue = FakeIssue([FakeComment('x' * 60_000), FakeComment('second')], comments_total=2)
        issue.body = 'd' * 19_500

        result = _make_client(issue).get_issue(42)

        assert result['comments'] == []
        assert 'too large to return' in result['comments_note']
        assert 'comments_offset=1' in result['comments_note']
        assert issue.html_url.rsplit('/', 1)[0] in result['comments_note']
        bounded, _ = bound_tool_result(result, 'get_issue', GITHUB_TOOLKIT_TYPE)
        assert len(bounded['body']) == 19_500, 'description was spent on a content-free comment'

    def test_a_tiny_body_is_never_replaced_by_a_longer_marker(self, bounding_limits):
        """The over-budget branch fires on the ~190-char metadata envelope alone, so it can
        reach a 4-char body. Replacing it with a 91-char marker grows the payload the
        shortener exists to shrink, and states a false length."""
        bounding_limits(enabled=True, limit=200_000, per_toolkit={GITHUB_TOOLKIT_TYPE: 20_000})
        for body in (None, '', 'ab', 'LGTM', 'thirteen char'):
            issue = FakeIssue([FakeComment(body), FakeComment('second')], comments_total=2)
            issue.body = 'd' * 19_500

            result = _make_client(issue).get_issue(42)

            assert result['comments'] == [], f'{body!r} was carried as a marker'
            assert 'truncated' not in (result.get('comments_note') or '')


def test_last_page_is_not_topped_up_with_a_wasted_fetch():
    """len(window) < LIMIT is always true for a nonzero offset; the page length is what
    says whether another page can exist."""
    issue = FakeIssue([FakeComment(f'comment {i}') for i in range(25)])

    result = _make_client(issue).get_issue(42, comments_offset=5)

    assert [c['body'] for c in result['comments']] == [f'comment {i}' for i in range(5, 25)]
    assert issue.paginated.page_fetches == 1


def test_shortened_comment_fits_the_budget_including_its_own_metadata():
    """Pinned on the helper, not end to end: the note reserve is larger than the metadata
    overhead, so it absorbs the difference and no whole-payload shape can tell the two apart."""
    projected = {
        'body': 'x' * 60_000,
        'user': 'alice',
        'created_at': '2026-01-03T10:05:00',
        'url': 'https://github.com/owner/repo/issues/42#issuecomment-1',
    }
    budget = 5_000

    projected['body'] = GitHubClient._shorten_comment_body(dict(projected), budget)

    assert estimate_chars(projected) <= budget
    assert '[truncated: 60000 chars' in projected['body']


def test_note_offers_a_direct_seek_to_the_end_of_the_thread():
    """Paging a long thread from the start cannot finish inside the default 25 steps, so
    the note must hand over the tail offset too."""
    total = 1642
    issue = FakeIssue([FakeComment(f'comment {i}') for i in range(GITHUB_ISSUE_COMMENTS_LIMIT)],
                      comments_total=total)

    note = _make_client(issue).get_issue(42)['comments_note']

    assert f'comments_offset={total - GITHUB_ISSUE_COMMENTS_LIMIT}' in note
    assert f'comments_offset={GITHUB_ISSUE_COMMENTS_LIMIT}' in note


def test_tail_offset_reaches_the_end_even_when_the_budget_shrank_the_window():
    """The tail offset must be derived from the window actually achieved, not from the
    nominal limit: a budget-shrunk window at total-30 lands 26 comments short of the end."""
    total = 1642
    big = 'x' * 50_000
    issue = FakeIssue([FakeComment(big) for _ in range(GITHUB_ISSUE_COMMENTS_LIMIT)],
                      comments_total=total)

    result = _make_client(issue).get_issue(42)
    window = len(result['comments'])
    tail = int(re.findall(r'comments_offset=(\d+)', result['comments_note'])[-1])

    assert window < GITHUB_ISSUE_COMMENTS_LIMIT
    assert tail + window >= total, (
        f'tail offset {tail} plus a {window}-comment window stops {total - tail - window} '
        f'short of {total}')


def test_tail_hint_is_withheld_when_it_points_at_what_the_caller_already_has():
    """For any thread of 31-60 comments the tail offset lands inside the window just
    returned, so naming it would send the model back over comments it already read."""
    for total in (31, 35, 51, 60):
        issue = FakeIssue([FakeComment(f'comment {i}')
                           for i in range(GITHUB_ISSUE_COMMENTS_LIMIT)], comments_total=total)

        note = _make_client(issue).get_issue(42)['comments_note']

        assert 'end of the thread' not in note, f'total={total}: {note}'
        assert f'comments_offset={GITHUB_ISSUE_COMMENTS_LIMIT}' in note


def test_toolkit_type_matches_the_registered_toolkit_name():
    """The budget resolves the limit by this string; a drift would silently read the
    global limit instead of the toolkit's override."""
    assert GITHUB_TOOLKIT_TYPE == GITHUB_TOOLKIT_NAME


def test_description_does_not_tell_the_model_to_page_until_the_note_clears():
    client = _make_client(FakeIssue([]))
    description = next(t['description'] for t in client.get_available_tools()
                       if t['name'] == 'get_issue')

    assert 'repeat until' not in description.lower()
    assert 'comments_offset' in description


def test_bound_tool_description_is_not_truncated_by_the_toolkit_cap():
    """get_tools caps every description at 1000 chars AFTER prepending Repository/Toolkit
    lines, so a long prompt loses its tail -- and the tail is the paging guidance. The
    registered description is pre-truncation, so only the bound tool shows this."""
    from elitea_sdk.tools.github import get_tools

    settings = {
        'selected_tools': ['get_issue'],
        'repository': 'a' * 39 + '/' + 'b' * 100,
        'active_branch': 'main',
        'base_branch': 'main',
        'github_configuration': {'access_token': 'x'},
    }
    tool = next(t for t in get_tools({'toolkit_name': 'x' * 128, 'settings': settings})
                if t.name == 'get_issue')

    from elitea_sdk.tools.github.tool_prompts import GET_ISSUE_PROMPT

    expected = (f"Repository: {settings['repository']}\n"
                f"Toolkit: {'x' * 128}\n"
                + GET_ISSUE_PROMPT.format(comments_limit=GITHUB_ISSUE_COMMENTS_LIMIT))

    assert tool.description == expected, (
        f'description was truncated: kept {len(tool.description)} of {len(expected)} chars. '
        f'get_tools caps it AFTER prepending the Repository/Toolkit lines, so the prompt must '
        f'be budgeted against the worst-case prefix, not a typical one'
    )


def test_unaligned_offset_tops_up_from_page_shape_not_the_comment_count():
    """Gating the top-up on comments_total silently under-fetched whenever that count was
    stale: real 59 / count 30 / offset 29 returned ONE comment of 30 available, with no note.
    Page shape is authoritative, so the top-up costs one extra fetch on an exactly-full last
    page -- deliberately paying a request to never lose a comment."""
    issue = FakeIssue([FakeComment(f'comment {i}') for i in range(60)])

    result = _make_client(issue).get_issue(42, comments_offset=35)

    assert [c['body'] for c in result['comments']] == [f'comment {i}' for i in range(35, 60)]
    assert issue.paginated.page_fetches == 2


def test_a_read_failure_past_the_end_reports_only_the_error():
    """comments_error and an overshoot note together would tell the model two different
    stories about why the window is empty."""
    issue = FakeIssue([], comments_total=3, raises=RuntimeError('502 Bad Gateway'))

    result = _make_client(issue).get_issue(42, comments_offset=99)

    assert result['comments'] == []
    assert '502 Bad Gateway' in result['comments_error']
    assert 'comments_note' not in result


def test_overshoot_never_names_an_offset_however_long_the_thread():
    """The branch only fires when window_start >= comments_total, and any last-window offset
    is comments_total - 30, so it is behind the caller by construction. Naming it let an
    all-withheld thread orbit between comments_total - 30 and comments_total forever."""
    for total in (1, 30, 31, 60, 137, 1642):
        issue = FakeIssue([FakeComment('c')], comments_total=total)

        note = _make_client(issue).get_issue(42, comments_offset=total + 9000)['comments_note']

        assert re.findall(r'comments_offset=(\d+)', note) == [], f'total={total}: {note}'
        assert str(total) in note


def test_every_offset_a_note_names_terminates(bounding_limits):
    """Walk every offset any note names, from every plausible start. A thread whose comments
    are all withheld walks 0->1->2->3->4 and then overshoots; suggesting 0 there closed a
    cycle back to the start."""
    bounding_limits(enabled=True, limit=200_000, per_toolkit={GITHUB_TOOLKIT_TYPE: 20_000})
    for total, start in [(4, 0), (4, 3), (4, 99), (30, 0), (31, 0), (60, 0), (137, 0), (137, 100)]:
        bodies = ['x' * 60_000] * total
        seen, offset, hops = [], start, 0
        while True:
            hops += 1
            assert hops <= total + 10, f'no termination, total={total} from {start}: {seen[-6:]}'
            issue = FakeIssue([FakeComment(b) for b in bodies], comments_total=total)
            issue.body = 'd' * 19_500
            note = _make_client(issue).get_issue(42, comments_offset=offset).get('comments_note')
            seen.append(offset)
            nxt = [int(m) for m in re.findall(r'comments_offset=(\d+)', note or '')]
            if not nxt:
                break
            assert nxt[0] not in seen, (
                f'cycle: total={total} from {start}, ...{seen[-4:]} -> revisit {nxt[0]}')
            offset = nxt[0]


ALREADY_OVERRUNNING_BEFORE_4957 = {
    'create_issue', 'update_issue', 'create_issue_on_project', 'update_issue_on_project',
    'get_commits', 'get_workflow_logs', 'get_me', 'search_code',
}


def test_no_new_tool_description_overruns_the_shared_cap():
    """get_tools caps EVERY bound description after prepending the Repository/Toolkit lines,
    so lengthening one prompt -- or lowering that shared constant -- silently truncates tools
    that have nothing to do with this change. Measured at the worst-case prefix: GitHub allows
    a 39-char owner and a 100-char repo, and the toolkit name column is String(128)."""
    from elitea_sdk.tools.github import get_tools

    settings = {
        'repository': 'a' * 39 + '/' + 'b' * 100,
        'active_branch': 'main',
        'base_branch': 'main',
        'github_configuration': {'access_token': 'x'},
    }
    tools = get_tools({'toolkit_name': 'x' * 128, 'settings': settings})
    truncated = {t.name for t in tools if len(t.description) >= 1000}

    assert 'get_issue' not in truncated
    assert truncated <= ALREADY_OVERRUNNING_BEFORE_4957, (
        f'newly truncated descriptions: {sorted(truncated - ALREADY_OVERRUNNING_BEFORE_4957)}. '
        f'This set is a ratchet: shrink it when a prompt is shortened, never widen it to admit '
        f'a new one'
    )


def test_shortener_withholds_at_the_exact_room_boundary():
    """room == 0 is the boundary between "cut it" and "cannot help"; off-by-one there puts
    back the bare-marker behaviour, where a 4-char body became a 91-char marker."""
    projected = {
        'body': 'x' * 60_000,
        'user': 'alice',
        'created_at': '2026-01-03T10:05:00',
        'url': 'https://github.com/owner/repo/issues/42#issuecomment-1',
    }
    marker = f"\n\n[truncated: {len(projected['body'])} chars, full text at {projected['url']}]"
    exact = estimate_chars({**projected, 'body': marker})

    assert GitHubClient._shorten_comment_body(dict(projected), exact) is None
    assert GitHubClient._shorten_comment_body(dict(projected), exact + 1) is not None


def test_note_reserve_covers_the_longest_note_the_code_emits():
    """The reserve is subtracted from the comment budget to pay for text added afterwards."""
    issue = FakeIssue([FakeComment(f'c{i}') for i in range(GITHUB_ISSUE_COMMENTS_LIMIT)],
                      comments_total=9_999_999)

    note = _make_client(issue).get_issue(42)['comments_note']

    assert 'Nearer the end' in note, 'expected the longest note variant'
    assert GITHUB_ISSUE_COMMENTS_NOTE_RESERVE >= estimate_chars({'comments_note': note}) + 64


class TestPromptStatesTheBehaviourItIsThereToDrive:
    """The prompt IS the delivery mechanism for the paging design, so its claims need pinning:
    each sentence below is load-bearing and was deletable with nothing going red."""

    def _description(self):
        client = _make_client(FakeIssue([]))
        return next(t['description'] for t in client.get_available_tools()
                    if t['name'] == 'get_issue')

    def test_states_that_an_absent_note_means_the_whole_thread(self):
        assert 'No note means' in self._description()

    def test_forbids_paging_a_long_thread_from_the_start(self):
        assert 'Never page' in self._description()

    def test_tells_the_model_to_take_the_skip_ahead_offset(self):
        assert 'skip-ahead offset' in self._description()

    def test_describes_the_second_offset_as_conditional(self):
        """"sometimes" is what stops the model relying on a tail offset that is often absent."""
        assert 'sometimes' in self._description()


def test_overshoot_note_tells_the_model_what_to_retry():
    issue = FakeIssue([FakeComment('c')], comments_total=5)

    note = _make_client(issue).get_issue(42, comments_offset=99)['comments_note']

    assert 'Retry below 5' in note


class TestStaleCommentCount:
    """issue.comments is a separate API read and goes stale the moment anyone comments, so
    nothing may depend on it for completeness. Every case here has a count that disagrees
    with the thread, in both directions."""

    def test_understated_count_does_not_suppress_the_note(self):
        """window_end >= comments_total, so only page shape can reveal the 30 more comments."""
        issue = FakeIssue([FakeComment(f'c{i}') for i in range(60)], comments_total=30)

        result = _make_client(issue).get_issue(42)

        assert len(result['comments']) == GITHUB_ISSUE_COMMENTS_LIMIT
        assert 'at least 30' in result['comments_note']
        assert 'comments_offset=30' in result['comments_note']

    def test_a_count_it_cannot_vouch_for_is_not_asserted_as_a_total(self):
        issue = FakeIssue([FakeComment(f'c{i}') for i in range(60)], comments_total=30)

        note = _make_client(issue).get_issue(42)['comments_note']

        assert 'of 30.' not in note, 'claimed a total the fetch contradicts'
        assert 'Nearer the end' not in note, 'tail offset derived from an untrustworthy count'

    def test_understated_count_does_not_under_fetch(self):
        """Gating the top-up on the count returned ONE comment of 30 available."""
        issue = FakeIssue([FakeComment(f'c{i}') for i in range(59)], comments_total=30)

        result = _make_client(issue).get_issue(42, comments_offset=29)

        assert [c['body'] for c in result['comments']] == [f'c{i}' for i in range(29, 59)]

    def test_overstated_count_with_an_in_range_offset_still_explains_itself(self):
        """The mirror race: a comment deleted between the issue read and the comments read."""
        issue = FakeIssue([], comments_total=5)

        result = _make_client(issue).get_issue(42, comments_offset=0)

        assert result['comments'] == []
        assert 'the issue reports 5' in result['comments_note']
        assert re.findall(r'comments_offset=(\d+)', result['comments_note']) == []

    def test_a_short_final_page_ends_the_thread_even_when_it_fills_the_shortfall(self):
        """`len(next_page) < shortfall` asks whether the page covered what we needed, not
        whether it was the last page. At total 31 offset 1 the shortfall is exactly 1 and the
        final page holds exactly 1, so the thread read as unfinished."""
        issue = FakeIssue([FakeComment(f'c{i}') for i in range(31)], comments_total=31)

        result = _make_client(issue).get_issue(42, comments_offset=1)

        assert [c['body'] for c in result['comments']] == [f'c{i}' for i in range(1, 31)]
        assert 'comments_note' not in result, 'claimed more remained after returning all of them'

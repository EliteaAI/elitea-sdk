"""Bounded, deterministic selection for large OpenAPI responses."""

from __future__ import annotations

import copy
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Optional


DEFAULT_RESPONSE_LIMIT = 50
MAX_RESPONSE_LIMIT = 200
MAX_SERIALIZED_RESPONSE_CHARS = 50_000
MAX_COLLECTION_DEPTH = 4
MAX_DISCOVERED_COLLECTIONS = 100
MAX_REPORTED_CANDIDATES = 20
COMMON_COLLECTION_KEYS = ("items", "results", "value", "data", "records")
BM25_K1 = 1.5
BM25_B = 0.75
PHRASE_SCORE_MULTIPLIER = 1.5
TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
QUERY_PART_PATTERN = re.compile(
    r'\s*(?P<negative>-)?(?:"(?P<phrase>(?:\\.|[^"\\])*)"|(?P<term>[^\s"]+))'
)


class ResponseSelectionError(ValueError):
    """Raised when response-selection controls are invalid."""


@dataclass(frozen=True)
class _Candidate:
    path: tuple[str, ...]
    collection: list[Any] | dict[str, Any]

    @property
    def kind(self) -> str:
        return "array" if isinstance(self.collection, list) else "object_map"

    @property
    def size(self) -> int:
        return len(self.collection)

    def search_documents(self) -> list[Any]:
        if isinstance(self.collection, list):
            return list(self.collection)
        return [[key, value] for key, value in self.collection.items()]

    def select(self, indices: Iterable[int]) -> list[Any] | dict[str, Any]:
        if isinstance(self.collection, list):
            return [self.collection[index] for index in indices]
        entries = list(self.collection.items())
        return {entries[index][0]: entries[index][1] for index in indices}

    def empty(self) -> list[Any] | dict[str, Any]:
        return [] if isinstance(self.collection, list) else {}


@dataclass(frozen=True)
class _SearchQuery:
    terms: tuple[str, ...] = ()
    phrases: tuple[tuple[str, ...], ...] = ()
    negative_terms: tuple[str, ...] = ()
    negative_phrases: tuple[tuple[str, ...], ...] = ()

    @property
    def enabled(self) -> bool:
        return bool(self.terms or self.phrases or self.negative_terms or self.negative_phrases)

    @property
    def scoring_terms(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.terms + tuple(term for phrase in self.phrases for term in phrase)))


@dataclass(frozen=True)
class _DocumentStats:
    source_index: int
    length: int
    term_frequencies: dict[str, int]
    eligible: bool
    phrase_hits: int


@dataclass(frozen=True)
class _RankedCandidate:
    candidate: _Candidate
    indices: list[int]
    top_score: float


def _resolve_local_ref(spec: dict[str, Any], ref: str) -> Optional[dict[str, Any]]:
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None

    current: Any = spec
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current if isinstance(current, dict) else None


def _resolve_schema_node(
    spec: dict[str, Any],
    node: Any,
    visited_refs: frozenset[str],
) -> tuple[Optional[dict[str, Any]], frozenset[str]]:
    if not isinstance(node, dict):
        return None, visited_refs
    ref = node.get("$ref")
    if not isinstance(ref, str):
        return node, visited_refs
    if ref in visited_refs:
        return None, visited_refs
    resolved = _resolve_local_ref(spec, ref)
    return resolved, visited_refs | {ref}


def _schema_collection_paths(
    spec: dict[str, Any],
    node: Any,
    *,
    path: tuple[str, ...] = (),
    depth: int = 0,
    visited_refs: frozenset[str] = frozenset(),
) -> list[tuple[str, ...]]:
    if depth > MAX_COLLECTION_DEPTH:
        return []

    schema, visited_refs = _resolve_schema_node(spec, node, visited_refs)
    if not schema:
        return []

    schema_type = schema.get("type")
    if schema_type == "array" or ("items" in schema and schema_type != "object"):
        return [path]
    if schema_type == "object" and (
        isinstance(schema.get("additionalProperties"), dict)
        or schema.get("additionalProperties") is True
    ):
        return [path]

    paths: list[tuple[str, ...]] = []
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, child in properties.items():
            if isinstance(name, str):
                paths.extend(
                    _schema_collection_paths(
                        spec,
                        child,
                        path=path + (name,),
                        depth=depth + 1,
                        visited_refs=visited_refs,
                    )
                )

    for composition_key in ("allOf", "oneOf", "anyOf"):
        branches = schema.get(composition_key)
        if isinstance(branches, list):
            for branch in branches:
                paths.extend(
                    _schema_collection_paths(
                        spec,
                        branch,
                        path=path,
                        depth=depth + 1,
                        visited_refs=visited_refs,
                    )
                )
    return paths


def get_response_collection_paths(
    spec: dict[str, Any], operation: dict[str, Any]
) -> list[tuple[str, ...]]:
    """Return collection paths declared by the first usable successful response schema."""
    responses = operation.get("responses") if isinstance(operation, dict) else None
    if not isinstance(responses, dict):
        return []

    success_responses = sorted(
        (
            (str(status), response)
            for status, response in responses.items()
            if str(status).startswith("2")
        ),
        key=lambda item: (item[0] != "200", item[0]),
    )
    for _, raw_response in success_responses:
        response, _ = _resolve_schema_node(spec, raw_response, frozenset())
        if not response:
            continue

        schemas: list[Any] = []
        content = response.get("content")
        if isinstance(content, dict):
            media_types = sorted(
                content,
                key=lambda media_type: (
                    str(media_type).lower() != "application/json",
                    "+json" not in str(media_type).lower(),
                    str(media_type),
                ),
            )
            for media_type in media_types:
                media = content.get(media_type)
                if isinstance(media, dict) and isinstance(media.get("schema"), dict):
                    schemas.append(media["schema"])
        if isinstance(response.get("schema"), dict):  # OpenAPI 2.0
            schemas.append(response["schema"])

        for schema in schemas:
            paths = _schema_collection_paths(spec, schema)
            if paths:
                return list(dict.fromkeys(paths))
    return []


def _tokenize(value: Any) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_PATTERN.finditer(str(value))]


def _parse_search(search: Optional[str]) -> _SearchQuery:
    if search is None or not str(search).strip():
        return _SearchQuery()

    text = str(search)
    terms: list[str] = []
    phrases: list[tuple[str, ...]] = []
    negative_terms: list[str] = []
    negative_phrases: list[tuple[str, ...]] = []
    position = 0

    while position < len(text):
        match = QUERY_PART_PATTERN.match(text, position)
        if not match:
            if not text[position:].strip():
                break
            raise ResponseSelectionError(
                f"Invalid response_search expression near: {text[position:position + 30]!r}"
            )
        if match.end() == position:
            raise ResponseSelectionError("Invalid response_search expression")
        position = match.end()

        negative = bool(match.group("negative"))
        phrase = match.group("phrase")
        raw_value = phrase if phrase is not None else match.group("term")
        if raw_value is None or raw_value == "-":
            raise ResponseSelectionError(
                "Invalid response_search expression: '-' must be followed by a word or phrase"
            )
        if phrase is not None:
            raw_value = raw_value.replace(r'\"', '"').replace("\\\\", "\\")
        parsed_tokens = tuple(_tokenize(raw_value))
        if not parsed_tokens:
            raise ResponseSelectionError(
                "response_search terms and phrases must contain letters or numbers"
            )

        if phrase is not None:
            (negative_phrases if negative else phrases).append(parsed_tokens)
        elif negative:
            negative_terms.extend(parsed_tokens)
        else:
            terms.extend(parsed_tokens)

    query = _SearchQuery(
        terms=tuple(dict.fromkeys(terms)),
        phrases=tuple(dict.fromkeys(phrases)),
        negative_terms=tuple(dict.fromkeys(negative_terms)),
        negative_phrases=tuple(dict.fromkeys(negative_phrases)),
    )
    if not query.enabled:
        raise ResponseSelectionError("response_search must contain at least one word or phrase")
    return query


def _value_token_data(value: Any) -> tuple[list[str], list[list[str]]]:
    tokens: list[str] = []
    sequences: list[list[str]] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                key_tokens = _tokenize(key)
                tokens.extend(key_tokens)
                if key_tokens:
                    sequences.append(key_tokens)
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)
        elif node is not None:
            value_tokens = _tokenize(node)
            tokens.extend(value_tokens)
            if value_tokens:
                sequences.append(value_tokens)

    visit(value)
    return tokens, sequences


def _contains_phrase(tokens: list[str], phrase: tuple[str, ...]) -> bool:
    phrase_length = len(phrase)
    if phrase_length == 0 or phrase_length > len(tokens):
        return False
    return any(
        tuple(tokens[index:index + phrase_length]) == phrase
        for index in range(len(tokens) - phrase_length + 1)
    )


def _sequences_contain_phrase(
    sequences: list[list[str]], phrase: tuple[str, ...]
) -> bool:
    return any(_contains_phrase(sequence, phrase) for sequence in sequences)


def _rank_values(values: list[Any], query: _SearchQuery) -> tuple[list[int], float]:
    if not query.enabled:
        return list(range(len(values))), 0.0

    scoring_terms = query.scoring_terms
    scoring_term_set = set(scoring_terms)
    document_frequencies: Counter[str] = Counter()
    documents: list[_DocumentStats] = []

    for source_index, value in enumerate(values):
        tokens, sequences = _value_token_data(value)
        token_counts = Counter(token for token in tokens if token in scoring_term_set)
        document_frequencies.update(token_counts.keys())
        token_set = set(tokens)
        excluded = any(term in token_set for term in query.negative_terms) or any(
            _sequences_contain_phrase(sequences, phrase)
            for phrase in query.negative_phrases
        )
        required_phrases_match = all(
            _sequences_contain_phrase(sequences, phrase) for phrase in query.phrases
        )
        has_scoring_match = not scoring_terms or any(token_counts.values())
        phrase_hits = sum(
            _sequences_contain_phrase(sequences, phrase) for phrase in query.phrases
        )
        documents.append(
            _DocumentStats(
                source_index=source_index,
                length=max(len(tokens), 1),
                term_frequencies=dict(token_counts),
                eligible=not excluded and required_phrases_match and has_scoring_match,
                phrase_hits=phrase_hits,
            )
        )

    if not documents:
        return [], 0.0

    document_count = len(documents)
    average_length = sum(document.length for document in documents) / document_count
    inverse_document_frequency = {
        term: math.log(
            1
            + (document_count - document_frequencies[term] + 0.5)
            / (document_frequencies[term] + 0.5)
        )
        for term in scoring_terms
    }
    phrase_boost = {
        phrase: PHRASE_SCORE_MULTIPLIER
        * sum(inverse_document_frequency.get(term, 0.0) for term in phrase)
        for phrase in query.phrases
    }

    ranked: list[tuple[float, int]] = []
    for document in documents:
        if not document.eligible:
            continue
        length_normalization = BM25_K1 * (
            1 - BM25_B + BM25_B * document.length / average_length
        )
        score = 0.0
        for term in scoring_terms:
            frequency = document.term_frequencies.get(term, 0)
            if frequency:
                score += inverse_document_frequency[term] * (
                    frequency * (BM25_K1 + 1) / (frequency + length_normalization)
                )
        if document.phrase_hits:
            score += sum(phrase_boost.values())
        ranked.append((score, document.source_index))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [item[1] for item in ranked], ranked[0][0] if ranked else 0.0


def _is_runtime_object_map(value: dict[str, Any]) -> bool:
    if len(value) < 3 or not all(
        isinstance(key, str) and isinstance(item, dict) and item
        for key, item in value.items()
    ):
        return False

    field_counts: Counter[str] = Counter(
        field
        for item in value.values()
        for field in item
        if isinstance(field, str)
    )
    return bool(field_counts) and max(field_counts.values()) / len(value) >= 0.7


def _discover_candidates(
    value: Any,
    preferred_paths: Iterable[tuple[str, ...]],
) -> list[_Candidate]:
    if isinstance(value, list):
        return [_Candidate((), value)]

    candidates: list[_Candidate] = []
    schema_paths = set(preferred_paths)

    def visit(node: Any, path: tuple[str, ...], depth: int) -> None:
        if (
            depth > MAX_COLLECTION_DEPTH
            or len(candidates) >= MAX_DISCOVERED_COLLECTIONS
            or not isinstance(node, dict)
        ):
            return
        has_schema_descendant = any(
            len(schema_path) > len(path) and schema_path[: len(path)] == path
            for schema_path in schema_paths
        )
        if path in schema_paths or (
            not has_schema_descendant and _is_runtime_object_map(node)
        ):
            candidates.append(_Candidate(path, node))
            return
        for key, child in node.items():
            if not isinstance(key, str):
                continue
            child_path = path + (key,)
            if isinstance(child, list):
                candidates.append(_Candidate(child_path, child))
            elif isinstance(child, dict):
                visit(child, child_path, depth + 1)

    visit(value, (), 0)
    return candidates


def _choose_candidate(
    candidates: list[_Candidate],
    preferred_paths: Iterable[tuple[str, ...]],
    query: _SearchQuery,
) -> Optional[_RankedCandidate]:
    if not candidates:
        return None
    if len(candidates) == 1:
        candidate = candidates[0]
        indices, top_score = _rank_values(candidate.search_documents(), query)
        return _RankedCandidate(candidate, indices, top_score)

    by_path = {candidate.path: candidate for candidate in candidates}
    schema_matches = [by_path[path] for path in preferred_paths if path in by_path]
    eligible_candidates = schema_matches or candidates

    ranked_candidates = [
        _RankedCandidate(candidate, *_rank_values(candidate.search_documents(), query))
        for candidate in eligible_candidates
    ]
    if query.scoring_terms:
        matching_candidates = [item for item in ranked_candidates if item.indices]
        if len(matching_candidates) == 1:
            return matching_candidates[0]
        if len(matching_candidates) > 1:
            matching_candidates.sort(key=lambda item: item.top_score, reverse=True)
            if not math.isclose(
                matching_candidates[0].top_score,
                matching_candidates[1].top_score,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                return matching_candidates[0]

    common = [
        candidate
        for candidate in eligible_candidates
        if candidate.path and candidate.path[-1].casefold() in COMMON_COLLECTION_KEYS
    ]
    if common:
        common.sort(
            key=lambda candidate: (
                COMMON_COLLECTION_KEYS.index(candidate.path[-1].casefold()),
                len(candidate.path),
                candidate.path,
            )
        )
        best = common[0]
        best_rank = (
            COMMON_COLLECTION_KEYS.index(best.path[-1].casefold()),
            len(best.path),
        )
        tied = [
            candidate
            for candidate in common
            if (
                COMMON_COLLECTION_KEYS.index(candidate.path[-1].casefold()),
                len(candidate.path),
            )
            == best_rank
        ]
        if len(tied) == 1:
            return next(item for item in ranked_candidates if item.candidate.path == best.path)

    by_size = sorted(eligible_candidates, key=lambda candidate: candidate.size, reverse=True)
    if len(by_size) == 1:
        return ranked_candidates[0]
    if by_size[0].size > 2 * by_size[1].size:
        return next(
            item for item in ranked_candidates if item.candidate.path == by_size[0].path
        )
    return None


def _format_path(path: tuple[str, ...]) -> str:
    if not path:
        return "$"
    formatted = "$"
    for part in path:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part):
            formatted += f".{part}"
        else:
            formatted += f"[{json.dumps(part, ensure_ascii=False)}]"
    return formatted


def _replace_collection(value: Any, path: tuple[str, ...], replacement: Any) -> Any:
    if not path:
        return replacement
    result = copy.deepcopy(value)
    current = result
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = replacement
    return result


def _serialize(metadata: dict[str, Any], data: Any) -> str:
    return json.dumps(
        {"_elitea_response_selection": metadata, "data": data},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _content_too_large(
    *,
    response_format: str,
    collection_path: str,
    collection_kind: str,
    total_items: int,
    matched_items: int,
    max_serialized_chars: int,
    query_enabled: bool,
) -> str:
    metadata = {
        "format": response_format,
        "collection_path": collection_path,
        "collection_kind": collection_kind,
        "total_items": total_items,
        "matched_items": matched_items,
        "returned_items": 0,
        "truncated": matched_items > 0,
        "ranking": "bm25" if query_enabled else "none",
        "result_order": "relevance" if query_enabled else "source",
        "status": "content_too_large",
        "message": "A selected item or response metadata exceeds the safe serialized response size.",
        "max_serialized_chars": max_serialized_chars,
    }
    return _serialize(metadata, None)


def _bounded_json_result(
    original: Any,
    candidate: _Candidate,
    matching_indices: list[int],
    limit: int,
    max_serialized_chars: int,
    query_enabled: bool,
) -> str:
    selected = candidate.select(matching_indices[:limit])
    path = _format_path(candidate.path)

    while selected:
        metadata = {
            "format": "json",
            "collection_path": path,
            "collection_kind": candidate.kind,
            "total_items": candidate.size,
            "matched_items": len(matching_indices),
            "returned_items": len(selected),
            "truncated": len(selected) < len(matching_indices),
            "ranking": "bm25" if query_enabled else "none",
            "result_order": "relevance" if query_enabled else "source",
        }
        output = _serialize(metadata, _replace_collection(original, candidate.path, selected))
        if len(output) <= max_serialized_chars:
            return output
        if isinstance(selected, list):
            selected.pop()
        else:
            selected.popitem()

    metadata = {
        "format": "json",
        "collection_path": path,
        "collection_kind": candidate.kind,
        "total_items": candidate.size,
        "matched_items": len(matching_indices),
        "returned_items": 0,
        "truncated": False,
        "ranking": "bm25" if query_enabled else "none",
        "result_order": "relevance" if query_enabled else "source",
    }
    empty_output = _serialize(
        metadata,
        _replace_collection(original, candidate.path, candidate.empty()),
    )
    if not matching_indices and len(empty_output) <= max_serialized_chars:
        return empty_output
    return _content_too_large(
        response_format="json",
        collection_path=path,
        collection_kind=candidate.kind,
        total_items=candidate.size,
        matched_items=len(matching_indices),
        max_serialized_chars=max_serialized_chars,
        query_enabled=query_enabled,
    )


def _bounded_text_result(
    segments: list[str],
    matches: list[str],
    limit: int,
    max_serialized_chars: int,
    query_enabled: bool,
) -> str:
    selected = list(matches[:limit])
    while selected:
        metadata = {
            "format": "text",
            "collection_path": "$segments",
            "collection_kind": "segments",
            "total_items": len(segments),
            "matched_items": len(matches),
            "returned_items": len(selected),
            "truncated": len(selected) < len(matches),
            "ranking": "bm25" if query_enabled else "none",
            "result_order": "relevance" if query_enabled else "source",
        }
        output = _serialize(metadata, "\n".join(selected))
        if len(output) <= max_serialized_chars:
            return output
        selected.pop()

    metadata = {
        "format": "text",
        "collection_path": "$segments",
        "collection_kind": "segments",
        "total_items": len(segments),
        "matched_items": len(matches),
        "returned_items": 0,
        "truncated": False,
        "ranking": "bm25" if query_enabled else "none",
        "result_order": "relevance" if query_enabled else "source",
    }
    empty_output = _serialize(metadata, "")
    if not matches and len(empty_output) <= max_serialized_chars:
        return empty_output
    return _content_too_large(
        response_format="text",
        collection_path="$segments",
        collection_kind="segments",
        total_items=len(segments),
        matched_items=len(matches),
        max_serialized_chars=max_serialized_chars,
        query_enabled=query_enabled,
    )


def select_response_content(
    content: str,
    *,
    response_search: Optional[str] = None,
    response_limit: Optional[int] = None,
    preferred_collection_paths: Iterable[tuple[str, ...]] = (),
    max_serialized_chars: int = MAX_SERIALIZED_RESPONSE_CHARS,
) -> str:
    """Select a bounded subset of a JSON collection or plain-text response."""
    query = _parse_search(response_search)
    if response_limit is None:
        limit = DEFAULT_RESPONSE_LIMIT
    elif isinstance(response_limit, bool) or not isinstance(response_limit, int):
        raise ResponseSelectionError("response_limit must be an integer")
    elif response_limit < 1 or response_limit > MAX_RESPONSE_LIMIT:
        raise ResponseSelectionError(
            f"response_limit must be between 1 and {MAX_RESPONSE_LIMIT}"
        )
    else:
        limit = response_limit

    if max_serialized_chars < 1:
        raise ResponseSelectionError("max_serialized_chars must be positive")

    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        text = str(content)
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        if len(paragraphs) > 1:
            segments = paragraphs
        else:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            segments = lines if len(lines) > 1 else paragraphs
        matching_indices, _ = _rank_values(segments, query)
        matches = [segments[index] for index in matching_indices]
        return _bounded_text_result(
            segments,
            matches,
            limit,
            max_serialized_chars,
            query.enabled,
        )

    preferred_collection_paths = tuple(preferred_collection_paths)
    candidates = _discover_candidates(parsed, preferred_collection_paths)
    ranked_candidate = _choose_candidate(candidates, preferred_collection_paths, query)
    if ranked_candidate is None:
        all_candidate_paths = sorted(_format_path(item.path) for item in candidates)
        candidate_paths = all_candidate_paths[:MAX_REPORTED_CANDIDATES]
        status = "ambiguous_collection" if all_candidate_paths else "collection_not_found"
        metadata = {
            "format": "json",
            "collection_path": None,
            "collection_kind": None,
            "total_items": None,
            "matched_items": None,
            "returned_items": 0,
            "truncated": False,
            "ranking": "bm25" if query.enabled else "none",
            "result_order": None,
            "status": status,
            "candidate_paths": candidate_paths,
            "candidate_count": len(all_candidate_paths),
            "message": (
                "Multiple response collections are equally plausible."
                if candidate_paths
                else "No response collection was found."
            ),
        }
        return _serialize(metadata, None)

    return _bounded_json_result(
        parsed,
        ranked_candidate.candidate,
        ranked_candidate.indices,
        limit,
        max_serialized_chars,
        query.enabled,
    )

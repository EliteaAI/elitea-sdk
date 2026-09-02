"""Toolkit-qualified tool selection and provider binding."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from langchain_core.tools import BaseTool
from pydantic import Field

from ..utils.constants import TOOLKIT_NAME_META, TOOLKIT_TYPE_META, TOOL_NAME_META


MAX_PROVIDER_TOOL_NAME_LENGTH = 128
_UNSAFE_TOOL_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]+")


@dataclass(frozen=True)
class ToolIdentity:
    toolkit_id: Any
    toolkit_name: Optional[str]
    toolkit_type: Optional[str]
    tool_name: str
    runtime_name: str

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            str(self.toolkit_id or ""),
            self.toolkit_name or "",
            self.toolkit_type or "",
            self.tool_name,
        )


class _QualifiedToolAlias(BaseTool):
    """Provider-visible alias that delegates execution to the original tool."""

    original_tool: BaseTool = Field(exclude=True)

    @staticmethod
    def _original_call(tool_input: Any, original_name: str) -> Any:
        if isinstance(tool_input, dict) and tool_input.get("type") == "tool_call":
            return {**tool_input, "name": original_name}
        return tool_input

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return self.original_tool.invoke(
            self._original_call(input, self.original_tool.name),
            config=config,
            **kwargs,
        )

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return await self.original_tool.ainvoke(
            self._original_call(input, self.original_tool.name),
            config=config,
            **kwargs,
        )

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return self.original_tool._run(*args, **kwargs)


@dataclass(frozen=True)
class ToolBindingPlan:
    provider_tools: list[BaseTool]
    tools_by_provider_name: dict[str, BaseTool]
    identities_by_provider_name: dict[str, ToolIdentity]

    def resolve(self, provider_name: str) -> Optional[BaseTool]:
        return self.tools_by_provider_name.get(provider_name)


def get_tool_identity(tool: BaseTool) -> ToolIdentity:
    metadata = getattr(tool, "metadata", None) or {}
    return ToolIdentity(
        toolkit_id=metadata.get("toolkit_id"),
        toolkit_name=metadata.get(TOOLKIT_NAME_META),
        toolkit_type=metadata.get(TOOLKIT_TYPE_META) or metadata.get("type"),
        tool_name=str(metadata.get(TOOL_NAME_META) or tool.name),
        runtime_name=str(tool.name),
    )


def select_tools_for_binding(
    tools: Sequence[BaseTool],
    selections: Mapping[str, Sequence[str]] | Sequence[str],
) -> tuple[list[BaseTool], list[str]]:
    """Resolve persisted tool selections without discarding toolkit identity."""

    available = [tool for tool in tools if isinstance(tool, BaseTool)]
    selected: list[BaseTool] = []
    selected_ids: set[int] = set()
    missing: list[str] = []

    def add(tool: BaseTool) -> None:
        if id(tool) not in selected_ids:
            selected.append(tool)
            selected_ids.add(id(tool))

    if isinstance(selections, Mapping):
        for toolkit_name, requested_names in selections.items():
            for requested_name in requested_names or []:
                matches = [
                    tool
                    for tool in available
                    if get_tool_identity(tool).toolkit_name == toolkit_name
                    and get_tool_identity(tool).tool_name == requested_name
                ]
                if len(matches) == 1:
                    add(matches[0])
                    continue
                if len(matches) > 1:
                    raise ValueError(
                        "Multiple tools match qualified identity "
                        f"'{toolkit_name}:{requested_name}'"
                    )

                # Compatibility for old tools that did not carry toolkit metadata.
                unscoped = [
                    tool
                    for tool in available
                    if not get_tool_identity(tool).toolkit_name
                    and get_tool_identity(tool).tool_name == requested_name
                ]
                if len(unscoped) == 1:
                    add(unscoped[0])
                else:
                    missing.append(f"{toolkit_name}:{requested_name}")
        return selected, missing

    for requested_name in selections or []:
        matches = [
            tool for tool in available
            if get_tool_identity(tool).tool_name == requested_name or tool.name == requested_name
        ]
        if not matches:
            missing.append(str(requested_name))
        for tool in matches:
            add(tool)
    return selected, missing


def build_tool_binding_plan(tools: Sequence[BaseTool]) -> ToolBindingPlan:
    """Create unique provider schemas while retaining exact execution targets."""

    originals = list(tools)
    identities = [get_tool_identity(tool) for tool in originals]
    # Existing runtime names may already carry a stable semantic prefix (for
    # example, ``sharepoint_search`` for the logical ``search`` operation).
    # Preserve those names and qualify only names that actually collide in the
    # provider schema.
    name_counts = Counter(identity.runtime_name for identity in identities)
    seen_identities: set[tuple[str, str, str, str]] = set()
    collision_aliases: dict[tuple[str, str, str, str], str] = {}
    reserved_names = {
        identity.runtime_name
        for identity in identities
        if name_counts[identity.runtime_name] == 1
    }

    for identity in identities:
        if name_counts[identity.runtime_name] <= 1:
            continue
        if identity.key in seen_identities:
            raise ValueError(f"Duplicate qualified tool identity: {identity.key}")
        seen_identities.add(identity.key)

        qualifier = identity.toolkit_name or identity.toolkit_type or identity.toolkit_id
        qualifier = _sanitize_name(qualifier)
        if not qualifier:
            raise ValueError(
                f"Cannot bind duplicate tool '{identity.tool_name}' without toolkit identity"
            )
        logical_name = _sanitize_name(identity.tool_name) or _sanitize_name(identity.runtime_name)
        collision_aliases[identity.key] = _bounded_name(
            f"{qualifier}__{logical_name}",
            identity,
        )

    alias_counts = Counter(collision_aliases.values())
    provider_tools: list[BaseTool] = []
    tools_by_provider_name: dict[str, BaseTool] = {}
    identities_by_provider_name: dict[str, ToolIdentity] = {}

    for tool, identity in zip(originals, identities):
        provider_name = identity.runtime_name
        description = tool.description

        if name_counts[identity.runtime_name] > 1:
            provider_name = collision_aliases[identity.key]
            if alias_counts[provider_name] > 1 or provider_name in reserved_names:
                provider_name = _bounded_name(provider_name, identity, force_hash=True)

            toolkit_label = identity.toolkit_name or str(
                identity.toolkit_id or identity.toolkit_type
            )
            toolkit_type = f" ({identity.toolkit_type})" if identity.toolkit_type else ""
            description = f"[Toolkit: {toolkit_label}{toolkit_type}] {description}"

        provider_tool = tool
        if provider_name != tool.name or description != tool.description:
            provider_tool = _QualifiedToolAlias(
                name=provider_name,
                description=description,
                args_schema=tool.args_schema,
                return_direct=tool.return_direct,
                metadata=tool.metadata,
                handle_tool_error=tool.handle_tool_error,
                handle_validation_error=tool.handle_validation_error,
                response_format=tool.response_format,
                original_tool=tool,
            )

        if provider_name in tools_by_provider_name:
            raise ValueError(
                f"Provider tool alias collision for qualified identity: {identity.key}"
            )
        provider_tools.append(provider_tool)
        tools_by_provider_name[provider_name] = tool
        identities_by_provider_name[provider_name] = identity

    return ToolBindingPlan(
        provider_tools=provider_tools,
        tools_by_provider_name=tools_by_provider_name,
        identities_by_provider_name=identities_by_provider_name,
    )


def _sanitize_name(value: Any) -> str:
    return _UNSAFE_TOOL_NAME_CHARS.sub("_", str(value or "")).strip("_-")


def _bounded_name(name: str, identity: ToolIdentity, force_hash: bool = False) -> str:
    if len(name) <= MAX_PROVIDER_TOOL_NAME_LENGTH and not force_hash:
        return name
    digest = hashlib.sha256("\x1f".join(identity.key).encode()).hexdigest()[:8]
    prefix_length = MAX_PROVIDER_TOOL_NAME_LENGTH - len(digest) - 2
    return f"{name[:prefix_length]}__{digest}"

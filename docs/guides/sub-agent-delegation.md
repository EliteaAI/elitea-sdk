# Sub-agent delegation

This guide describes the agent-agnostic Task toolkit pattern: how to delegate
work from a parent agent to specialized sub-agents using the `Application` tool.
This is the same pattern used by Anthropic's Task tool, deepagents' `task` tool,
and VS Code's subagent harness.

## When to use it

Use sub-agent delegation when you need to:

- Hand off a focused subtask to an agent with a different system prompt and a
  curated tool set (e.g. a "research" agent vs. a "writer" agent).
- Keep the orchestrator's context clean by isolating sub-agent reasoning.
- Compose multiple specialists rather than fitting everything into one agent.

Use **swarm mode** (`langgraph_swarm`) instead when you need peer-to-peer
handoffs with shared message state. Use **nested pipelines** when the work is
deterministic and graph-shaped rather than agent-driven.

## How it works

`Application` is a `BaseTool` with a `task: str` argument schema. Attach an
existing platform agent or pipeline as a child by adding it to the parent's
toolkits — `ApplicationToolkit.get_toolkit(...)` builds the `Application` tool
from `application_id` / `application_version_id` and stores its display
metadata for chip rendering.

When the orchestrator LLM emits a `tool_call` for that Application:

1. The child agent runs in an **isolated context**. It receives only the
   `task` string and any agent variables defined on its toolkit; it does not
   see the parent's `state["messages"]`.
2. The child's `thread_id` is derived from the parent's thread_id and the
   `tool_call_id`: `f"{parent_thread_id}:{tool_call_id}"`. Same `tool_call_id`
   resolves to the same child checkpoint, so HITL resume is stable. Different
   `tool_call_id`s isolate concurrent or repeated invocations.
3. The child returns an `AgentResponse` (`output`, `messages`, `thread_id`,
   `execution_finished`, `hitl_interrupt`, `context_info`, `hitl_decisions`).
4. The parent's `ToolMessage.content` is collapsed to the child's `output`
   string. The orchestrator LLM sees a clean conversational result and reasons
   over it on the next turn.

## The `task` contract

The orchestrator LLM is told (via the `TASK_DELEGATION_ADDON` injected when
agent tools are present):

- Inline ALL context the sub-agent needs in `task`. No anaphora.
- Delegate independent subtasks; do simple work yourself.
- Each sub-agent returns a separate ToolMessage; reason over results in the
  next turn.
- You may emit multiple sub-agent tool_calls in one response; they are
  processed in order.

The "no anaphora" rule is the most important: the child cannot read the
parent's conversation, so phrases like "do that thing again" or "continue from
where we left off" will fail. Always inline the data the child needs.

## Failure isolation

If a sub-agent raises, the parent receives an error `ToolMessage` (not a
propagated exception) and the LLM is re-invoked so it can react. This is
covered by the existing per-tool exception handling in
`__perform_tool_calling` — siblings continue executing.

## Sequential today, parallel later

Multiple sub-agent tool_calls in one assistant response currently execute
**in order**. Concurrent fan-out is filed as a follow-up enhancement that
requires a multi-interrupt HITL contract (so two children both hitting
sensitive-tool guards can each be approved independently). The system prompt
addon will switch from "processed in order" to "for parallel execution" once
that ships. Until then, the LLM is free to emit multiple tool_calls per turn —
they just serialize.

## Related

- `elitea_sdk/runtime/tools/application.py` — the `Application` `BaseTool` and
  context-quarantine logic.
- `elitea_sdk/runtime/toolkits/application.py` — the `ApplicationToolkit` that
  wires platform agents/pipelines into an `Application` tool.
- `elitea_sdk/runtime/models/agent_response.py` — the `AgentResponse` contract.
- `elitea_sdk/runtime/langchain/constants.py` — `TASK_DELEGATION_ADDON`.

# LangGraph Capped Tool-Call History Fix

## Reason

When the graph hits `MAX_CODEX_TOOL_TURNS`, it currently exits to `final_response` with the latest `AIMessage.tool_calls` still unhandled. The checkpointer keeps that message in the thread, and the `messages` reducer appends the next user turn onto it. The next model request can then contain an assistant tool call without a matching `ToolMessage`.

## Plan

- Add a regression test that exhausts the tool-turn cap, starts another turn, and validates tool-call history shape.
- Add a capped-tool-call node that appends synthetic `ToolMessage`s for unexecuted calls and sets a final response.
- Route capped tool calls to that node instead of `final_response`.
- Run focused LangGraph tests and nearby processor/tool tests.

# No Deterministic Text-To-Motion Fallback

Agent Orchestration must not synthesize MoveIt action calls from loose user-text substrings after a model tool-call failure. We prefer the model/tool loop to call a tool, explain a concrete blocker, or ask for clarification; deterministic robot code stays focused on the Task Policy Layer, Robot Call Validation, Robot Context, and MoveIt execution shape. This avoids hidden language assumptions such as treating every command containing "up", "down", or "wave" as permission to move.

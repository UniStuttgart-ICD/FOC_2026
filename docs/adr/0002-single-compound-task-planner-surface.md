# Single Task-Level Manipulation Planner Surface

Agent Orchestration exposes one model-visible task-level manipulation planner, `moveit_plan_manipulation_task`, for goals such as `hold`, `release`, `move_and_release`, and `pick_place`. The near-term backend is explicit staged MoveIt composition, not MTC and not a silent fallback. A future MTC backend may replace or sit behind the same planner surface after it can solve and preview the same contracts reliably.

`moveit_plan_pick_task`, `moveit_plan_place_task`, and the MTC-shaped `moveit_plan_compound_task` may remain MCP/manual internals during migration, but they are hidden from the Pipecat model-visible tool surface so the agent does not choose between competing task-planning contracts. This keeps preview, approval, execution-contract, and release semantics in one place at the cost of making the task-level planner responsible for ordinary manipulation workflows.

Follow-up: long-running manipulation planning should move behind Robot Job Blackboard planner jobs. A planner worker should run deterministic planner progression until it returns either a complete Task Solution or a terminal failure, without asking the LLM to drive intermediate free-motion/cartesian stages.

# Simulation-First Dual Task Execution

Status: Accepted

Agent Orchestration exposes one model-visible task execution tool, `moveit_execute_task`, for approved `task_solution_id` values. Execution always targets MoveIt/RViz first and also attempts Verified Real Robot Execution when connected; RViz success is task execution success, while real robot status is reported separately.

This is not a fallback from real robot execution. RViz is an intended execution target so AR testing remains usable when hardware is off, and the old `moveit_execute_task_solution`, `moveit_execute_task_plan`, and `moveit_execute_plan` paths remain hidden/internal compatibility tools instead of competing model-visible choices.

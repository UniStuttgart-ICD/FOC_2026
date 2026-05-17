# Canonical manipulation examples
Assume each manipulation example starts by observing relevant robot and scene state.

User: "Kibbitz, pick up dynamic_3"
- Call moveit_list_scene_objects and use dynamic_3 only if it is one returned object_name.
- Call moveit_get_object_context for dynamic_3.
- Call moveit_plan_manipulation_task with backend="staged_moveit", requirements.goal="hold", requirements.object_name="dynamic_3", and bounded requirements.lift_distance_m.
- Execute only the returned task_solution_id with moveit_execute_task_plan when execution is explicitly approved.

User: "Kibbitz, let go"
- If the current held object is fresh and clear, call moveit_plan_manipulation_task with backend="staged_moveit" and requirements.goal="release".
- If the held object is stale or unclear, observe first. If it is still unclear, ask which object should be released.

User: "Kibbitz, move up" / "wave to me" / "draw a short line"
- These are free-space motion requests, not manipulation tasks. Do not fake them through moveit_plan_manipulation_task.
- Ask for a supported object task or use the AR free/cartesian controls outside the model-visible manipulation surface.

User: "Kibbitz, bring me that"
- Use fresh user sensing to resolve "that"; if gaze, manual target, or scene object context is stale or unclear, ask which object the user means.
- If user sensing shows gaze object candidate dynamic_5, call moveit_list_scene_objects and use dynamic_5 only if it is one returned object_name.
- Call moveit_get_object_context for the chosen object and use the returned grasp-relevant faces and ground-plane clearance.
- Call moveit_plan_manipulation_task with backend="staged_moveit", requirements.goal="pick_place", requirements.object_name, and a target pose from Geometry World Context or fresh user-position standoff context.
- Use the fresh user position as human destination context with about 0.40 m standoff from the human instead of the exact user position.
- If the current tool list cannot complete the pickup or delivery safely, explain the blocker briefly; do not pretend the pickup or delivery happened.

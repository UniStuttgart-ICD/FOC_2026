# Canonical manipulation examples
Assume each manipulation example starts by observing relevant robot and scene state.

User: "Kibbitz, pick up dynamic_3"
- Call moveit_list_scene_objects and use dynamic_3 only if it is one returned object_name.
- Call moveit_get_object_context for dynamic_3.
- Call moveit_plan_manipulation_task with requirements.goal="hold", requirements.object_name="dynamic_3", and bounded requirements.lift_distance_m.
- Execute only the returned task_solution_id with moveit_execute_task when execution is explicitly approved.

User: "Kibbitz, pick up element 1 from the top"
- Call moveit_list_scene_objects and use dynamic_1 only if it is one returned object_name.
- Call moveit_get_object_context for dynamic_1 and verify "top" is one returned raw.object.grasp_faces[].name.
- Call moveit_plan_manipulation_task with requirements.goal="hold", requirements.object_name="dynamic_1", requirements.grasp_face="top", and bounded requirements.lift_distance_m.
- If planning fails for the required top face, explain that blocker briefly and ask before trying another face.
- Execute only the returned task_solution_id with moveit_execute_task when execution is explicitly approved.

User: "Kibbitz, let go"
- If the current held object is fresh and clear, call moveit_plan_manipulation_task with requirements.goal="release".
- If the held object is stale or unclear, observe first. If it is still unclear, ask which object should be released.

User: "Kibbitz, place element 2 there"
- "There" means the matching Geometry World Context target pose without saying hologram.
- Call moveit_list_scene_objects and use dynamic_2 only if it is one returned object_name.
- Call moveit_get_object_context for dynamic_2 and check whether the object is free or already attached.
- If dynamic_2 is free, call moveit_plan_manipulation_task with requirements.goal="pick_place", requirements.object_name="dynamic_2", and requirements.target_pose from Geometry World Context.
- If dynamic_2 is already held or attached, call moveit_plan_manipulation_task with requirements.goal="move_and_release", requirements.object_name="dynamic_2", and requirements.target_pose from Geometry World Context.
- If Geometry World Context is blocked or lacks a valid target_pose for dynamic_2, ask for an updated target instead of inferring one.

User: "Kibbitz, move up" / "wave to me" / "draw a short line"
- These are free-space motion requests, not manipulation tasks. Do not fake them through moveit_plan_manipulation_task.
- Ask for a supported object task or use the AR free/cartesian controls outside the model-visible manipulation surface.

User: "Kibbitz, bring me that"
- Use fresh user sensing to resolve "that"; if gaze, manual target, or scene object context is stale or unclear, ask which object the user means.
- If user sensing shows gaze object candidate dynamic_5, call moveit_list_scene_objects and use dynamic_5 only if it is one returned object_name.
- Call moveit_get_object_context for the chosen object and use the returned grasp-relevant faces and ground-plane clearance.
- If the chosen object is free, call moveit_plan_manipulation_task with requirements.goal="pick_place", requirements.object_name, and a target pose from fresh Vizor user position standoff context.
- If the chosen object is already held or attached, call moveit_plan_manipulation_task with requirements.goal="move_and_release", requirements.object_name, and a target pose from fresh Vizor user position standoff context.
- Use the fresh Vizor user position as human destination context with about 0.40 m standoff from the human instead of the exact user position.
- If the current tool list cannot complete the pickup or delivery safely, explain the blocker briefly; do not pretend the pickup or delivery happened.

User: "Kibbitz, bring element 2 to me"
- Call moveit_list_scene_objects and use dynamic_2 only if it is one returned object_name.
- Call moveit_get_object_context for dynamic_2 and check whether the object is free or already attached.
- Derive the target object pose from the fresh Vizor user position with about 0.40 m standoff from the human.
- If dynamic_2 is free, call moveit_plan_manipulation_task with requirements.goal="pick_place", requirements.object_name="dynamic_2", and the derived target pose.
- If dynamic_2 is already held or attached, call moveit_plan_manipulation_task with requirements.goal="move_and_release", requirements.object_name="dynamic_2", and the derived target pose.

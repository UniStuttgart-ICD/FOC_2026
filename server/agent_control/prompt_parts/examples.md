# Canonical manipulation examples
Assume each manipulation example starts by observing relevant robot and scene state.

User: "Kibbitz, pick up dynamic_3"
- Call moveit_list_scene_objects and use dynamic_3 only if it is one returned object_name.
- Call moveit_get_object_context for dynamic_3.
- Call moveit_plan_manipulation_task with requirements.goal="hold", requirements.object_name="dynamic_3", and bounded requirements.lift_distance_m.
- Execute only the returned task_solution_id with moveit_execute_task when execution is explicitly approved.

User: "Kibbitz, hold element 2"
- Call moveit_list_scene_objects and use dynamic_2 only if it is one returned object_name.
- Call moveit_get_object_context for dynamic_2.
- Call moveit_plan_manipulation_task with requirements.goal="hold", requirements.object_name="dynamic_2", and requirements.lift_distance_m=0.0.
- Execute only the returned task_solution_id with moveit_execute_task when execution is explicitly approved.

User: "Kibbitz, pick up element 1 from the top"
- Call moveit_list_scene_objects and use dynamic_1 only if it is one returned object_name.
- Call moveit_get_object_context for dynamic_1 and verify "top" is one returned raw.object.grasp_faces[].name.
- Call moveit_plan_manipulation_task with requirements.goal="hold", requirements.object_name="dynamic_1", requirements.grasp_face="top", and bounded requirements.lift_distance_m.
- If planning fails for the required top face, explain that blocker briefly and ask before trying another face.
- Execute only the returned task_solution_id with moveit_execute_task when execution is explicitly approved.

User: "Kibbitz, good, just move 20 cm to your body"
- Treat this as a move-only held-object request if the current object is held or attached.
- Call moveit_plan_manipulation_task with requirements.goal="move" and motion-only human_relative or relative_tcp requirements that preserve orientation and keep holding.
- This is not requirements.goal="move_and_release"; do not call a release/place goal unless the user explicitly asks to release, place, or deliver.
- Execute only the returned task_solution_id with moveit_execute_task when execution is explicitly approved.

User: "Kibbitz, try to go just up 30 cm"
- Treat this as a motion-only TCP move.
- Call moveit_get_current_pose first.
- Call moveit_plan_manipulation_task with requirements.goal="move", requirements.motion.type="relative_tcp", requirements.motion.direction="up", and requirements.motion.distance_m=0.30.
- The move preserves current TCP orientation and does not open the gripper, detach, release, or place anything.
- Execute only the returned task_solution_id with moveit_execute_task when execution is explicitly approved.

User: "Kibbitz, come closer to me"
- Treat this as a motion-only human-relative TCP move.
- Call moveit_get_current_pose first and use fresh Vizor user position.
- Call moveit_plan_manipulation_task with requirements.goal="move", requirements.motion.type="human_relative", requirements.motion.relation="toward_user", and a bounded requirements.motion.distance_m.
- If fresh Vizor user position is missing or stale, ask for clarification instead of guessing.

User: "Kibbitz, go away from me"
- Treat this as a motion-only human-relative TCP move.
- Call moveit_get_current_pose first and use fresh Vizor user position.
- Call moveit_plan_manipulation_task with requirements.goal="move", requirements.motion.type="human_relative", requirements.motion.relation="away_from_user", and a bounded requirements.motion.distance_m.
- If fresh Vizor user position is missing or stale, ask for clarification instead of guessing.

User: "Kibbitz, let go"
- If the current held object is fresh and clear, call moveit_plan_manipulation_task with requirements.goal="release".
- If the held object is stale or unclear, observe first. If it is still unclear, ask which object should be released.

User: "Kibbitz, hold element 2, then release it"
- First call moveit_list_scene_objects and use dynamic_2 only if it is one returned object_name.
- Call moveit_get_object_context for dynamic_2.
- Plan the hold with moveit_plan_manipulation_task using requirements.goal="hold", requirements.object_name="dynamic_2", and requirements.lift_distance_m=0.0.
- After hold execution and fresh held-object proof, the explicit release intent may be planned with requirements.goal="release".

User: "Kibbitz, place element 2 there"
- "There" means the matching Geometry World Context target pose without saying hologram.
- Call moveit_list_scene_objects and use dynamic_2 only if it is one returned object_name.
- Call moveit_get_object_context for dynamic_2 and check whether the object is free or already attached.
- If dynamic_2 is free, call moveit_plan_manipulation_task with requirements.goal="pick_place", requirements.object_name="dynamic_2", and requirements.target_pose from Geometry World Context.
- If dynamic_2 is already held or attached, call moveit_plan_manipulation_task with requirements.goal="move_and_release", requirements.object_name="dynamic_2", and requirements.target_pose from Geometry World Context.
- If Geometry World Context is blocked or lacks a valid target_pose for dynamic_2, ask for an updated target instead of inferring one.

User: "Kibbitz, move it there and release it"
- Require fresh held-object context before planning.
- Because this has explicit release intent, call moveit_plan_manipulation_task with requirements.goal="move_and_release", requirements.object_name for the held object, and requirements.target_pose from Geometry World Context.
- Execute only the returned task_solution_id with moveit_execute_task when execution is explicitly approved.

User: "Kibbitz, pick element 2 and place it there"
- "There" means the matching Geometry World Context target pose without saying hologram.
- Call moveit_list_scene_objects and use dynamic_2 only if it is one returned object_name.
- Call moveit_get_object_context for dynamic_2 and verify it is free.
- If dynamic_2 is free, call moveit_plan_manipulation_task with requirements.goal="pick_place", requirements.object_name="dynamic_2", and requirements.target_pose from Geometry World Context.
- Execute only the returned task_solution_id with moveit_execute_task when execution is explicitly approved.

User: "Kibbitz, wave to me" / "draw a short line"
- These are expressive multi-waypoint free-space paths, not the motion-only move goal.
- They are unsupported through the model-visible manipulation surface unless a supported move/hold/release/place task is requested.

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

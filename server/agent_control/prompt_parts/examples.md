# Canonical motion examples
Assume each example starts by calling moveit_get_current_pose for a fresh pose, then preserving the current orientation in each target pose.

User: "Kibbitz, move up"
- If the fresh TCP pose is x=0.57, y=0.39, z=0.62, call moveit_plan_free_motion with target_pose x=0.57, y=0.39, z=0.82.
- Execute only the returned raw.plan_name with moveit_execute_plan when execution is explicitly requested; say `Hmmmmmm. Moved up 200 mm with the robot.` only after execution verification passes.

User: "Kibbitz, wave to me"
- Use moveit_plan_cartesian_motion with waypoints near the fresh pose: lift 0.15 m, move left 0.20 m, move right 0.20 m, and return near center.
- This is a visible 40 cm side-to-side wave. Preserve the current orientation.

User: "Kibbitz, draw a short line"
- Use moveit_plan_cartesian_motion with a visible line near the fresh pose, such as y-0.20 m to y+0.20 m at the current z, preserving orientation.

User: "Kibbitz, draw a small circle"
- Use moveit_plan_cartesian_motion with clear waypoints around the fresh pose, preserving orientation.

User: "Kibbitz, bring me that"
- Use fresh user sensing to resolve "that"; if gaze, manual target, or scene object context is stale or unclear, ask which object the user means.
- If user sensing shows gaze object candidate dynamic_5, call moveit_list_scene_objects and use dynamic_5 only if it is one returned object_name.
- Call moveit_get_object_context for the chosen object and choose an approach from the returned grasp-relevant faces and ground-plane clearance.
- Call moveit_plan_pick for the chosen object; use the returned raw.plan_name only after the user explicitly confirms execution.
- After execution, call moveit_verify_attached_object before saying the object was picked up or moved with the gripper.
- Use the fresh user position as the human destination context, but plan any TCP waypoint with about 0.40 m standoff from the human instead of at the exact user position.
- If the current tool list cannot complete the pickup or delivery safely, explain the blocker briefly; do not pretend the pickup or delivery happened.

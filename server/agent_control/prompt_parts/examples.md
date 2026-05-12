# Canonical motion examples
Assume each example starts by calling moveit_get_current_pose for a fresh pose, then preserving the current orientation in each target pose.

User: "Kibbitz, move up"
- If the fresh TCP pose is x=0.57, y=0.39, z=0.62, call moveit_plan_and_execute_free_motion with target_pose x=0.57, y=0.39, z=0.82.
- Say `Hmmmmmm. Moved up 200 mm with the robot.` after the tool succeeds.

User: "Kibbitz, wave to me"
- Use moveit_plan_and_execute_cartesian_motion with waypoints near the fresh pose: lift 0.15 m, move left 0.20 m, move right 0.20 m, and return near center.
- This is a visible 40 cm side-to-side wave. Preserve the current orientation.

User: "Kibbitz, draw a short line"
- Use moveit_plan_and_execute_cartesian_motion with a visible line near the fresh pose, such as y-0.20 m to y+0.20 m at the current z, preserving orientation.

User: "Kibbitz, draw a small circle"
- Use moveit_plan_and_execute_cartesian_motion with bounded waypoints around the fresh pose, about 0.18 m radius, preserving orientation.

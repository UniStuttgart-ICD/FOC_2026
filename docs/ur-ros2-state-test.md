# UR ROS 2 Humble Real Robot State Test

Status on 2026-05-27: the isolated ROS 2 Humble sidecar is implemented and live-tested against the real UR10e at `169.254.130.206`.

## Scope

This sidecar is for real-robot state and handshake verification only. It is separate from `workshop.compose.yml` and the ROS 1 workshop runtime.

Run it from the repo root:

```powershell
docker compose -f ur-ros2.compose.yml up --build
```

Use commands inside the sidecar like this:

```powershell
docker compose -f ur-ros2.compose.yml exec ur-ros2-driver bash -lc "source /ur_ws/install/setup.bash && ros2 node list"
```

## Current Launch Contract

The sidecar launches:

```text
ros2 launch ur_robot_driver ur_control.launch.py
  ur_type:=ur10e
  robot_ip:=169.254.130.206
  reverse_ip:=169.254.130.5
  headless_mode:=true
  activate_joint_controller:=false
  launch_rviz:=false
```

Runtime settings:

- Docker host networking, no Compose `ports:`.
- `ROS_DOMAIN_ID=42`.
- `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`.
- ROS 2 driver and UR client library are source-built in the image to avoid the Humble binary client-library 1 second configuration timeout.

## Operator Setup

Put the robot in Remote Control mode before starting the sidecar. Clear any emergency stop, power on, and brake-release the robot before resending the headless program.

Do not press Play for the current headless setup. ROS sends the External Control script directly.

## Verified Live State

The live acceptance check reached this state:

- Dashboard safety mode: `NORMAL`.
- Dashboard program running: `true`.
- Remote Control: `true`.
- `/joint_states` publishes once.
- `ros2 node list` and `ros2 topic list` return the expected driver graph.
- `ros2 control list_controllers` is readable.
- Reverse, trajectory, and script command sockets stay connected.
- Motion controllers, including `scaled_joint_trajectory_controller`, are inactive.

Useful checks:

```powershell
docker compose -f ur-ros2.compose.yml exec ur-ros2-driver bash -lc "source /ur_ws/install/setup.bash && ros2 topic echo /joint_states --once"
docker compose -f ur-ros2.compose.yml exec ur-ros2-driver bash -lc "source /ur_ws/install/setup.bash && ros2 control list_controllers"
docker compose -f ur-ros2.compose.yml exec ur-ros2-driver bash -lc "source /ur_ws/install/setup.bash && ros2 service call /dashboard_client/get_safety_mode ur_dashboard_msgs/srv/GetSafetyMode '{}'"
docker compose -f ur-ros2.compose.yml exec ur-ros2-driver bash -lc "source /ur_ws/install/setup.bash && ros2 service call /dashboard_client/program_running ur_dashboard_msgs/srv/IsProgramRunning '{}'"
```

## Motion Gate

Do not use this sidecar for robot motion yet.

Reasons:

- The launch intentionally uses `activate_joint_controller:=false`.
- The driver reports a calibration mismatch for this physical robot.
- No MoveIt 2 planning or execution path has been added to this sidecar.

Before commanding motion:

1. Extract and pass the real robot calibration with `ur_calibration`.
2. Decide the command frame for Cartesian moves, such as base-frame Z up or tool-frame up.
3. Enable the intended joint trajectory controller explicitly.
4. Plan and execute through MoveIt 2 or a reviewed trajectory action path.
5. Keep an operator at the pendant and emergency stop.

## Troubleshooting Notes

- If the driver cannot read the robot configuration quickly enough, use the source-built image; the pinned UR client library waits 10 seconds.
- If the pendant shows trajectory or script command socket connection failures, verify `reverse_ip:=169.254.130.5` and Remote Control mode.
- If the reverse interface connects and then drops during a state-only run, verify `activate_joint_controller:=false`.
- If the robot reports `ROBOT_EMERGENCY_STOP`, clear the physical safety state before resending the program.

from __future__ import annotations

import py_compile
import subprocess
from pathlib import Path


ROBOT_PY = Path("/root/catkin_ws/src/vizor_lib/src/vizor_lib/robot.py")
CATKIN_SRC = Path("/root/catkin_ws/src")
OLD_JUMP_LINE = "            jump_threshold = 0.0 #0.15\n"
OLD_CALL = (
    "                result = self.move_group.compute_cartesian_path("
    "target_poses, eef_step, jump_threshold) #default avoid collision"
)
NEW_CALL = """                result = self.move_group.compute_cartesian_path(
                    target_poses,
                    eef_step,
                    avoid_collisions=True,
                ) #default avoid collision"""
OLD_WORLD_POSE = """    pose.position.z = z
    worldXY.pose = pose
"""
NEW_WORLD_POSE = """    pose.position.z = z
    pose.orientation.w = 1.0
    worldXY.pose = pose
"""
OLD_WORLD_FRAME = '    header.frame_id = "base" # GoFa\n'
NEW_WORLD_FRAME = '    header.frame_id = "base_link"\n'
CARTESIAN_SUBSCRIBER = (
    '        rospy.Subscriber(f"{self.name}/request/cartesian", '
    "PlanningCartesian, self.planCartesianMotion, callback_args=self.name)\n"
)
SAMPLED_SUBSCRIBER = (
    '        rospy.Subscriber(f"{self.name}/request/sampled", '
    "PlanningCartesian, self.planSampledApproachMotion, callback_args=self.name)\n"
)
EXECUTE_SUBSCRIBER = (
    '        rospy.Subscriber(f"{self.name}/command/execute", '
    "String, self.executeStoredMotion, callback_args=self.name)\n"
)
STOP_AGENT_PATH_SUBSCRIBER = (
    '        rospy.Subscriber(f"{self.name}/command/stop", '
    "String, self.stopAgentPath, callback_args=self.name)\n"
)
CARTESIAN_METHOD = "    def planCartesianMotion(self, msg, *args):\n"
EXECUTE_STORED_METHOD = "    def executeStoredMotion(self, msg, *args):\n"
AGENT_PATH_METHODS = '''    def _agent_path_stage_names(self):
        return sorted(
            name
            for name in self.trajectory_data.keys()
            if isinstance(name, str) and name.startswith("AgentPath:")
        )

    def executeAgentPath(self, name):
        stage_names = self._agent_path_stage_names()
        if not stage_names:
            print("AgentPath unavailable; re-observe and replan")
            return
        self.active_agent_path = {"name": name, "stages": stage_names, "requires_task_bridge": True}
        print("AgentPath execution requires the verified task bridge; refusing loose stage execution")

    def stopAgentPath(self, msg, *args):
        if msg.data != "AgentPath":
            return
        self.active_agent_path = None
        try:
            self.move_group.stop()
        except Exception as e:
            print(e)
        for stage_name in self._agent_path_stage_names():
            self.trajectory_data.pop(stage_name, None)

'''
SAMPLED_METHODS = '''    def _combine_sampled_segments(self, segments):
        if not segments:
            return None
        if len(segments) == 1:
            return segments[0]
        combined = copy.deepcopy(segments[0])
        for segment in segments[1:]:
            points = segment.joint_trajectory.points
            if not points:
                continue
            offset = combined.joint_trajectory.points[-1].time_from_start if combined.joint_trajectory.points else rospy.Duration(0)
            for point in points[1:]:
                shifted = copy.deepcopy(point)
                shifted.time_from_start = shifted.time_from_start + offset
                combined.joint_trajectory.points.append(shifted)
        return combined

    def planSampledApproachMotion(self, msg, *args):
        name = args[0]
        self.move_group.set_planning_pipeline_id("ompl")
        self.move_group.set_planner_id("RRTConnect")
        print (f"planning sampled approach {msg.name}")
        if DEBUG:
            print (f"   from {self._get_current_pose()}")
            print (f"   through {msg.poses}")
        rospy.sleep(DELAY)
        try:
            target_poses = msg.poses
            if not target_poses:
                self.planning_status_publisher.publish(String("planning failed"))
                return
            segments = []
            self.move_group.set_start_state_to_current_state()
            for target_pose in target_poses:
                result = self.move_group.plan(joints = target_pose)
                if not result[0]:
                    print(f"planning failed {result[3]}")
                    code = int(str(result[3]).split(':')[-1])
                    self.planning_status_publisher.publish(String(f"{self.planningResponseForHumans(code)}"))
                    return
                output = result[1]
                segments.append(output)
                if output.joint_trajectory.points:
                    next_state = self.robot.get_current_state()
                    next_state.joint_state.name = output.joint_trajectory.joint_names
                    next_state.joint_state.position = output.joint_trajectory.points[-1].positions
                    self.move_group.set_start_state(next_state)
            output = self._combine_sampled_segments(segments)
            if output:
                if DEBUG: print (f"   >> sampled approach movement with {len(output.joint_trajectory.points)}")
                convertedTraj = output if not self.need_offset else self.joint_offset_func(output) #moveit_msgs/RobotTrajectory
                if msg.name == "hololens_path_sampled" or self.store_plan:
                    with open(f'{self.root}/{name}/{msg.name}.yaml', 'w') as fp:
                        yaml.dump(output, fp, default_flow_style=True)
                self._publish_trajectory(convertedTraj.joint_trajectory, msg.name)
                if msg.name in self.trajectory_data.keys():
                    print (f"warn: overwriting previous trajectory {msg.name}")
                self.trajectory_data[msg.name] = output
                self.planning_status_publisher.publish(String("success"))
            else:
                self.planning_status_publisher.publish(String("planning failed"))
        except Exception as e:
            print(e)
        finally:
            self.move_group.set_start_state_to_current_state()
            self.move_group.set_planning_pipeline_id("pilz_industrial_motion_planner")

'''
ROBOTIQ_COLLISION_BLOCK = """    <disable_collisions link1="robotiq_arg2f_base_link" link2="wrist_1_link" reason="Never"/>
    <disable_collisions link1="robotiq_arg2f_base_link" link2="wrist_2_link" reason="Never"/>
    <disable_collisions link1="robotiq_arg2f_base_link" link2="wrist_3_link" reason="Adjacent"/>
    <disable_collisions link1="left_outer_knuckle" link2="robotiq_arg2f_base_link" reason="Adjacent"/>
    <disable_collisions link1="right_outer_knuckle" link2="robotiq_arg2f_base_link" reason="Adjacent"/>
    <disable_collisions link1="left_inner_knuckle" link2="robotiq_arg2f_base_link" reason="Adjacent"/>
    <disable_collisions link1="right_inner_knuckle" link2="robotiq_arg2f_base_link" reason="Adjacent"/>
    <disable_collisions link1="left_outer_finger" link2="left_outer_knuckle" reason="Adjacent"/>
    <disable_collisions link1="right_outer_finger" link2="right_outer_knuckle" reason="Adjacent"/>
    <disable_collisions link1="left_inner_finger" link2="left_outer_finger" reason="Adjacent"/>
    <disable_collisions link1="right_inner_finger" link2="right_outer_finger" reason="Adjacent"/>
    <disable_collisions link1="left_inner_knuckle" link2="left_inner_finger" reason="Never"/>
    <disable_collisions link1="right_inner_knuckle" link2="right_inner_finger" reason="Never"/>
    <disable_collisions link1="left_inner_finger_pad" link2="left_inner_finger" reason="Adjacent"/>
    <disable_collisions link1="right_inner_finger_pad" link2="right_inner_finger" reason="Adjacent"/>
    <disable_collisions link1="left_inner_finger_pad" link2="right_inner_finger_pad" reason="Never"/>
    <disable_collisions link1="left_outer_finger" link2="right_outer_finger" reason="Never"/>
    <disable_collisions link1="left_inner_finger" link2="right_inner_finger" reason="Never"/>
    <disable_collisions link1="left_inner_finger_pad" link2="wrist_3_link" reason="Never"/>
    <disable_collisions link1="right_inner_finger_pad" link2="wrist_3_link" reason="Never"/>"""
ROBOTIQ_COLLISION_LINES = frozenset(ROBOTIQ_COLLISION_BLOCK.splitlines())
MTC_EXECUTE_TASK_SOLUTION_CAPABILITY = "move_group/ExecuteTaskSolutionCapability"


def _write_if_changed(path: Path, text: str) -> bool:
    if path.read_text() == text:
        return False
    path.write_text(text)
    return True


def patch_vizor_robot_py() -> bool:
    text = ROBOT_PY.read_text()
    changed = False

    if OLD_CALL in text:
        text = text.replace(OLD_JUMP_LINE, "", 1)
        text = text.replace(OLD_CALL, NEW_CALL, 1)
        changed = True
    elif "avoid_collisions=True" not in text:
        raise SystemExit(f"Expected old compute_cartesian_path call not found in {ROBOT_PY}")

    if OLD_WORLD_FRAME in text:
        text = text.replace(OLD_WORLD_FRAME, NEW_WORLD_FRAME, 1)
        changed = True
    elif 'header.frame_id = "base_link"' not in text:
        raise SystemExit(f"Expected base world frame line not found in {ROBOT_PY}")

    if "pose.orientation.w = 1.0" not in text:
        if OLD_WORLD_POSE not in text:
            raise SystemExit(f"Expected create_world_pose body not found in {ROBOT_PY}")
        text = text.replace(OLD_WORLD_POSE, NEW_WORLD_POSE, 1)
        changed = True

    if SAMPLED_SUBSCRIBER not in text:
        if CARTESIAN_SUBSCRIBER not in text:
            raise SystemExit(f"Expected cartesian request subscriber not found in {ROBOT_PY}")
        text = text.replace(CARTESIAN_SUBSCRIBER, f"{CARTESIAN_SUBSCRIBER}{SAMPLED_SUBSCRIBER}", 1)
        changed = True

    if STOP_AGENT_PATH_SUBSCRIBER not in text:
        if EXECUTE_SUBSCRIBER in text:
            text = text.replace(EXECUTE_SUBSCRIBER, f"{EXECUTE_SUBSCRIBER}{STOP_AGENT_PATH_SUBSCRIBER}", 1)
            changed = True

    if "self.active_agent_path" not in text and "self.trajectory_data = {}" in text:
        text = text.replace("self.trajectory_data = {}\n", "self.trajectory_data = {}\n        self.active_agent_path = None\n", 1)
        changed = True

    if "def planSampledApproachMotion" not in text:
        if CARTESIAN_METHOD not in text:
            raise SystemExit(f"Expected planCartesianMotion method not found in {ROBOT_PY}")
        text = text.replace(CARTESIAN_METHOD, f"{SAMPLED_METHODS}{CARTESIAN_METHOD}", 1)
        changed = True

    if "def executeAgentPath" not in text:
        if EXECUTE_STORED_METHOD in text:
            text = text.replace(EXECUTE_STORED_METHOD, f"{AGENT_PATH_METHODS}{EXECUTE_STORED_METHOD}", 1)
            changed = True

    if 'if msg.data == "AgentPath":' not in text:
        if EXECUTE_STORED_METHOD in text:
            text = text.replace(
                EXECUTE_STORED_METHOD,
                f'{EXECUTE_STORED_METHOD}        if msg.data == "AgentPath":\n'
                "            self.executeAgentPath(msg.data)\n"
                "            return\n",
                1,
            )
            changed = True

    if changed:
        ROBOT_PY.write_text(text)
    py_compile.compile(str(ROBOT_PY), doraise=True)
    return changed


def patch_robotiq_2f85_integration(src_root: Path = CATKIN_SRC) -> bool:
    changed = False
    ur10_xacro = src_root / "urdf_support" / "ur10_support" / "urdf" / "ur10_with_gripper.xacro"
    srdf = src_root / "moveit_support" / "ur10_moveit_support" / "config" / "ur10_with_gripper.srdf"
    load_robot = src_root / "vizor_package" / "launch" / "load_robot.launch"
    vizor2ros = src_root / "vizor_package" / "launch" / "vizor2ros.launch"
    robotiq_launch = src_root / "robotiq_2f_gripper_control" / "launch" / "robotiq_action_server.launch"

    changed |= _write_if_changed(
        ur10_xacro,
        """<?xml version="1.0"?>
<robot name="ur10_with_gripper" xmlns:xacro="http://wiki.ros.org/xacro">
  <xacro:include filename="$(find ur10_support)/urdf/ur10_macro.xacro"/>
  <xacro:include filename="$(find robotiq_2f_85_gripper_visualization)/urdf/robotiq_arg2f_85_model_macro.xacro"/>

  <xacro:ur10_robot prefix=""/>
  <xacro:robotiq_arg2f_85 prefix=""/>

  <joint name="tool0_to_robotiq_arg2f_base_link" type="fixed">
    <parent link="tool0"/>
    <child link="robotiq_arg2f_base_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <link name="tcp"/>
  <joint name="robotiq_arg2f_base_link_to_tcp" type="fixed">
    <parent link="robotiq_arg2f_base_link"/>
    <child link="tcp"/>
    <origin xyz="0 0 0.138" rpy="0 0 0"/>
  </joint>
</robot>
""",
    )

    srdf_text = srdf.read_text()
    srdf_lines = [
        line
        for line in srdf_text.splitlines()
        if "gripper_base" not in line
        and "gripper_fingers" not in line
        and "robotiq_arg2f_base_link" not in line
        and line not in ROBOTIQ_COLLISION_LINES
    ]
    srdf_text = "\n".join(srdf_lines) + "\n"
    marker = '    <virtual_joint name="virtual_joint" type="fixed" parent_frame="world" child_link="base_link"/>'
    if marker not in srdf_text:
        raise SystemExit(f"Expected virtual joint marker not found in {srdf}")
    srdf_text = srdf_text.replace(marker, f"{marker}\n{ROBOTIQ_COLLISION_BLOCK}", 1)
    changed |= _write_if_changed(srdf, srdf_text)

    load_text = load_robot.read_text()
    if 'name="use_robotiq_2f85"' not in load_text:
        load_text = load_text.replace(
            '  <arg name="moveit_pkg_name" default="crb15000_moveit_support"/>\n',
            '  <arg name="moveit_pkg_name" default="crb15000_moveit_support"/>\n'
            '  <arg name="use_robotiq_2f85" default="false"/>\n'
            '  <arg name="robotiq_sim" default="true"/>\n'
            '  <arg name="robotiq_comport" default="/dev/ttyUSB0"/>\n'
            '  <arg name="robotiq_baud" default="115200"/>\n',
            1,
        )
    load_text = load_text.replace(
        "[move_group/fake_controller_joint_states]",
        "[move_group/fake_controller_joint_states, gripper_joint_states]",
    )
    if "robotiq_action_server.launch" not in load_text:
        robotiq_include = """
    <include file="$(find robotiq_2f_gripper_control)/launch/robotiq_action_server.launch" if="$(arg use_robotiq_2f85)">
      <arg name="sim" value="$(arg robotiq_sim)"/>
      <arg name="comport" value="$(arg robotiq_comport)"/>
      <arg name="baud" value="$(arg robotiq_baud)"/>
      <arg name="joint_name" value="finger_joint"/>
      <arg name="stroke" value="0.085"/>
    </include>
"""
        if "    <!-- Fake Execution -->" in load_text:
            load_text = load_text.replace("    <!-- Fake Execution -->", f"{robotiq_include}\n    <!-- Fake Execution -->", 1)
        else:
            load_text = load_text.replace("  </group>", f"{robotiq_include}\n  </group>", 1)
    changed |= _write_if_changed(load_robot, load_text)

    vizor_text = vizor2ros.read_text()
    if 'name="use_robotiq_2f85"' not in vizor_text:
        robotiq_args = (
            '  <arg name="use_robotiq_2f85" default="true"/>\n'
            '  <arg name="robotiq_sim" default="true"/>\n'
            '  <arg name="robotiq_comport" default="/dev/ttyUSB0"/>\n'
            '  <arg name="robotiq_baud" default="115200"/>\n'
        )
        if '  <arg name="physical" default="false"/>\n' in vizor_text:
            vizor_text = vizor_text.replace(
                '  <arg name="physical" default="false"/>\n',
                f'  <arg name="physical" default="false"/>\n{robotiq_args}',
                1,
            )
        else:
            vizor_text = vizor_text.replace("<launch>\n", f"<launch>\n{robotiq_args}", 1)
    if '<arg name="use_robotiq_2f85" value="$(arg use_robotiq_2f85)"/>' not in vizor_text:
        vizor_text = vizor_text.replace(
            '      <arg name="use_rviz" value="$(arg use_rviz)" />\n',
            '      <arg name="use_rviz" value="$(arg use_rviz)" />\n'
            '      <arg name="use_robotiq_2f85" value="$(arg use_robotiq_2f85)"/>\n'
            '      <arg name="robotiq_sim" value="$(arg robotiq_sim)"/>\n'
            '      <arg name="robotiq_comport" value="$(arg robotiq_comport)"/>\n'
            '      <arg name="robotiq_baud" value="$(arg robotiq_baud)"/>\n',
            1,
        )
    changed |= _write_if_changed(vizor2ros, vizor_text)

    robotiq_text = robotiq_launch.read_text()
    if '<remap from="/joint_states" to="gripper_joint_states"/>' not in robotiq_text:
        robotiq_text = robotiq_text.replace(
            '        <param name="comport" value="$(arg comport)" />',
            '        <remap from="/joint_states" to="gripper_joint_states"/>\n'
            '        <param name="comport" value="$(arg comport)" />',
            1,
        )
        if '<remap from="/joint_states" to="gripper_joint_states"/>' not in robotiq_text:
            robotiq_text = robotiq_text.replace(
                '        <param name="joint_name" value="$(arg joint_name)" />',
                '        <remap from="/joint_states" to="gripper_joint_states"/>\n'
                '        <param name="joint_name" value="$(arg joint_name)" />',
                1,
            )
    changed |= _write_if_changed(robotiq_launch, robotiq_text)
    return changed


def patch_mtc_move_group_capability(src_root: Path = CATKIN_SRC) -> bool:
    load_robot = src_root / "vizor_package" / "launch" / "load_robot.launch"
    load_text = load_robot.read_text()

    if "move_group.launch" not in load_text:
        return False

    capability_arg = '<arg name="capabilities" value="$(arg move_group_capabilities)"/>'
    if capability_arg not in load_text:
        lines = load_text.splitlines()
        for index, line in enumerate(lines):
            if "move_group.launch" in line and not line.rstrip().endswith("/>"):
                indent = line[: len(line) - len(line.lstrip())] + "  "
                lines.insert(index + 1, f"{indent}{capability_arg}")
                load_text = "\n".join(lines) + "\n"
                break
        else:
            return False

    if 'name="move_group_capabilities"' not in load_text:
        load_text = load_text.replace(
            '  <arg name="moveit_pkg_name" default="crb15000_moveit_support"/>\n',
            '  <arg name="moveit_pkg_name" default="crb15000_moveit_support"/>\n'
            f'  <arg name="move_group_capabilities" default="{MTC_EXECUTE_TASK_SOLUTION_CAPABILITY}"/>\n',
            1,
        )
        if 'name="move_group_capabilities"' not in load_text:
            load_text = load_text.replace(
                "<launch>\n",
                f'<launch>\n  <arg name="move_group_capabilities" default="{MTC_EXECUTE_TASK_SOLUTION_CAPABILITY}"/>\n',
                1,
            )

    return _write_if_changed(load_robot, load_text)


def regenerate_ur10_with_gripper_urdf() -> None:
    subprocess.run(
        [
            "bash",
            "-lc",
            "source /opt/ros/noetic/setup.bash && "
            "export ROS_PACKAGE_PATH=/root/catkin_ws/src:${ROS_PACKAGE_PATH:-} && "
            "xacro /root/catkin_ws/src/urdf_support/ur10_support/urdf/ur10_with_gripper.xacro "
            "> /root/catkin_ws/src/urdf_support/ur10_support/urdf/ur10_with_gripper.urdf",
        ],
        check=True,
    )


def main() -> None:
    robot_changed = patch_vizor_robot_py()
    robotiq_changed = patch_robotiq_2f85_integration()
    mtc_changed = patch_mtc_move_group_capability()
    regenerate_ur10_with_gripper_urdf()
    print(f"Patched {ROBOT_PY}" if robot_changed else f"{ROBOT_PY} already patched")
    print("Patched Robotiq 2F-85 integration" if robotiq_changed else "Robotiq 2F-85 integration already patched")
    print("Patched MTC move_group capability" if mtc_changed else "MTC move_group capability already patched or no hook found")


if __name__ == "__main__":
    main()

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
OLD_PILZ_PIPELINE = 'self.move_group.set_planning_pipeline_id("pilz_industrial_motion_planner")'
NEW_OMPL_PIPELINE = 'self.move_group.set_planning_pipeline_id("ompl")'
OLD_PILZ_PTP_PLANNER = 'self.move_group.set_planner_id("PTP")'
OLD_PILZ_LIN_PLANNER = 'self.move_group.set_planner_id("LIN")'
NEW_RRTCONNECT_PLANNER = 'self.move_group.set_planner_id("RRTConnect")'
OLD_GROUND_PLANE_POSES = (
    "        ground_pose = create_world_pose(z = -0.01)\n",
    "        ground_pose = create_world_pose(z = -0.07)\n",
    "        ground_pose = create_world_pose(z = -0.02)\n",
    "        ground_pose = create_world_pose(z = -0.08)\n",
    "        ground_pose = create_world_pose(z = -0.095)\n",
)
NEW_GROUND_PLANE_POSE = "        ground_pose = create_world_pose(z = -0.105)\n"
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
PLANNING_LOG_IMPORTS = "import json\nimport time\nimport traceback\n"
CARTESIAN_METHOD = "    def planCartesianMotion(self, msg, *args):\n"
FREE_METHOD = "    def planFreeMotion(self, msg, *args):\n"
COMBINE_SAMPLED_METHOD = "    def _combine_sampled_segments(self, segments):\n"
PLAN_TRANSITION_METHOD = "    def plan_transition(self, joint_trajectory_point, name = \"transition\"):\n"
EXECUTE_STORED_METHOD = "    def executeStoredMotion(self, msg, *args):\n"
PUBLISH_TRAJECTORY_METHOD = "    def _publish_trajectory(self, joint_trajectory, traj_name):\n"
AR_PREVIEW_TIME_SCALE_LINE = "AR_PREVIEW_TIME_SCALE = 4.0\n"
AR_PREVIEW_METHODS = '''    def _ar_preview_trajectory(self, joint_trajectory):
        preview = copy.deepcopy(joint_trajectory)
        for point in preview.points:
            point.time_from_start = rospy.Duration.from_sec(point.time_from_start.to_sec() * AR_PREVIEW_TIME_SCALE)
            if point.velocities:
                point.velocities = [value / AR_PREVIEW_TIME_SCALE for value in point.velocities]
            if point.accelerations:
                point.accelerations = [value / (AR_PREVIEW_TIME_SCALE * AR_PREVIEW_TIME_SCALE) for value in point.accelerations]
        return preview

'''
RAW_PUBLISH_TRAJECTORY_ASSIGNMENT = "        traj.joint_trajectory = joint_trajectory\n"
AR_PREVIEW_PUBLISH_TRAJECTORY_ASSIGNMENT = (
    "        traj.joint_trajectory = self._ar_preview_trajectory(joint_trajectory)\n"
)
PLANNING_LOG_METHODS = '''    def _moveit_plan_tuple_value(self, result, index):
        try:
            return result[index]
        except Exception:
            return None

    def _moveit_error_code_to_int(self, value):
        try:
            return int(str(value).split(":")[-1])
        except Exception:
            return None

    def _moveit_pose_to_dict(self, pose):
        if pose is None:
            return None
        position = getattr(pose, "position", None)
        orientation = getattr(pose, "orientation", None)
        if position is None or orientation is None:
            return str(pose)
        return {
            "position": {
                "x": getattr(position, "x", None),
                "y": getattr(position, "y", None),
                "z": getattr(position, "z", None),
            },
            "orientation": {
                "x": getattr(orientation, "x", None),
                "y": getattr(orientation, "y", None),
                "z": getattr(orientation, "z", None),
                "w": getattr(orientation, "w", None),
            },
        }

    def _moveit_pose_list_to_dicts(self, poses):
        return [self._moveit_pose_to_dict(pose) for pose in poses or []]

    def _moveit_joint_state_to_dict(self, point):
        if point is None:
            return None
        return {
            "positions": list(getattr(point, "positions", []) or []),
            "velocities": list(getattr(point, "velocities", []) or []),
            "accelerations": list(getattr(point, "accelerations", []) or []),
        }

    def _moveit_trajectory_point_count(self, trajectory):
        try:
            return len(trajectory.joint_trajectory.points)
        except Exception:
            return 0

    def _safe_current_pose_log(self):
        try:
            return self._moveit_joint_state_to_dict(self._get_current_pose())
        except Exception as e:
            return {"error": str(e)}

    def _moveit_known_scene_objects(self):
        try:
            return list(self.scene.get_known_object_names())
        except Exception as e:
            return {"error": str(e)}

    def _write_moveit_planning_log(self, record):
        try:
            log_path = os.environ.get("MOVEIT_PLANNING_LOG_PATH")
            if not log_path:
                return
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            payload = {
                "schema": "moveit_planning_diagnostics.v1",
                "timestamp_unix": time.time(),
                "robot_name": self.name,
            }
            payload.update(record)
            with open(log_path, "a") as fp:
                fp.write(json.dumps(payload, default=str, sort_keys=True) + "\\n")
        except Exception as e:
            print(f"planning diagnostics log failed: {e}")

'''
FREE_METHOD_LOGGED = '''    def planFreeMotion(self, msg, *args):
        # rospy.wait_for_service('plan_free_motion')
        name = args[0]
        self.move_group.set_planning_pipeline_id("ompl")
        self.move_group.set_planner_id("RRTConnect")
        print (f"planning free {msg.name}")
        if DEBUG:
            print (f"   from {self._get_current_pose()}")
            print (f"   to {msg.target_pose}")
        rospy.sleep(DELAY)
        try:
            target_pose = msg.target_pose
            start_pose = self._safe_current_pose_log()
            known_scene_objects = self._moveit_known_scene_objects()
            result = self.move_group.plan(joints = target_pose)
            output = self._moveit_plan_tuple_value(result, 1)
            self._write_moveit_planning_log({
                "request_type": "free",
                "request_topic": f"{self.name}/request/free",
                "plan_name": msg.name,
                "planner_pipeline": "ompl",
                "planner_id": "RRTConnect",
                "current_pose": start_pose,
                "target_pose": self._moveit_pose_to_dict(target_pose),
                "target_poses": [self._moveit_pose_to_dict(target_pose)],
                "known_scene_objects": known_scene_objects,
                "success": bool(self._moveit_plan_tuple_value(result, 0)),
                "planning_time": self._moveit_plan_tuple_value(result, 2),
                "moveit_error_code": self._moveit_error_code_to_int(self._moveit_plan_tuple_value(result, 3)),
                "trajectory_points": self._moveit_trajectory_point_count(output),
            })
            if result[0]:
                output = result[1]
                if DEBUG: print (f"   >> planning time {result[2]}")
                print (f"   >> rrtconnect movement with {len(output.joint_trajectory.points)}")
                convertedTraj = output if not self.need_offset else self.joint_offset_func(output) #moveit_msgs/RobotTrajectory

                if msg.name == "hololens_path_free" or self.store_plan:
                    with open(f'{self.root}/{name}/{msg.name}.yaml', 'w') as fp:
                        yaml.dump(output, fp, default_flow_style=True)

                self._publish_trajectory(convertedTraj.joint_trajectory, msg.name)

                if msg.name in self.trajectory_data.keys():
                    print (f"warn: overwriting previous trajectory {msg.name}")
                self.trajectory_data[msg.name] = output

            else:
                print(f"planning failed {result[3]}")
            code = int(str(result[3]).split(':')[-1])
            self.planning_status_publisher.publish(String(f"{self.planningResponseForHumans(code)}"))
        except Exception as e:
            self._write_moveit_planning_log({
                "request_type": "free",
                "request_topic": f"{self.name}/request/free",
                "plan_name": getattr(msg, "name", None),
                "planner_pipeline": "ompl",
                "planner_id": "RRTConnect",
                "success": False,
                "exception": str(e),
                "traceback": traceback.format_exc(),
            })
            print(e)

'''
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
            start_pose = self._safe_current_pose_log()
            known_scene_objects = self._moveit_known_scene_objects()
            if not target_poses:
                self._write_moveit_planning_log({
                    "request_type": "sampled",
                    "request_topic": f"{self.name}/request/sampled",
                    "plan_name": msg.name,
                    "planner_pipeline": "ompl",
                    "planner_id": "RRTConnect",
                    "current_pose": start_pose,
                    "target_poses": [],
                    "known_scene_objects": known_scene_objects,
                    "success": False,
                    "status": "planning failed",
                    "trajectory_points": 0,
                })
                self.planning_status_publisher.publish(String("planning failed"))
                return
            segments = []
            self.move_group.set_start_state_to_current_state()
            for segment_index, target_pose in enumerate(target_poses):
                result = self.move_group.plan(joints = target_pose)
                output = self._moveit_plan_tuple_value(result, 1)
                self._write_moveit_planning_log({
                    "request_type": "sampled",
                    "request_topic": f"{self.name}/request/sampled",
                    "plan_name": msg.name,
                    "planner_pipeline": "ompl",
                    "planner_id": "RRTConnect",
                    "sampled_segment_index": segment_index,
                    "current_pose": start_pose,
                    "target_pose": self._moveit_pose_to_dict(target_pose),
                    "target_poses": self._moveit_pose_list_to_dicts(target_poses),
                    "known_scene_objects": known_scene_objects,
                    "success": bool(self._moveit_plan_tuple_value(result, 0)),
                    "planning_time": self._moveit_plan_tuple_value(result, 2),
                    "moveit_error_code": self._moveit_error_code_to_int(self._moveit_plan_tuple_value(result, 3)),
                    "trajectory_points": self._moveit_trajectory_point_count(output),
                })
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
                self._write_moveit_planning_log({
                    "request_type": "sampled",
                    "request_topic": f"{self.name}/request/sampled",
                    "plan_name": msg.name,
                    "planner_pipeline": "ompl",
                    "planner_id": "RRTConnect",
                    "sampled_segment_index": "combined",
                    "current_pose": start_pose,
                    "target_poses": self._moveit_pose_list_to_dicts(target_poses),
                    "known_scene_objects": known_scene_objects,
                    "success": True,
                    "trajectory_points": self._moveit_trajectory_point_count(output),
                })
                self.planning_status_publisher.publish(String("success"))
            else:
                self._write_moveit_planning_log({
                    "request_type": "sampled",
                    "request_topic": f"{self.name}/request/sampled",
                    "plan_name": msg.name,
                    "planner_pipeline": "ompl",
                    "planner_id": "RRTConnect",
                    "sampled_segment_index": "combined",
                    "current_pose": start_pose,
                    "target_poses": self._moveit_pose_list_to_dicts(target_poses),
                    "known_scene_objects": known_scene_objects,
                    "success": False,
                    "status": "planning failed",
                    "trajectory_points": 0,
                })
                self.planning_status_publisher.publish(String("planning failed"))
        except Exception as e:
            self._write_moveit_planning_log({
                "request_type": "sampled",
                "request_topic": f"{self.name}/request/sampled",
                "plan_name": getattr(msg, "name", None),
                "planner_pipeline": "ompl",
                "planner_id": "RRTConnect",
                "success": False,
                "exception": str(e),
                "traceback": traceback.format_exc(),
            })
            print(e)
        finally:
            self.move_group.set_start_state_to_current_state()

'''
CARTESIAN_METHOD_LOGGED = '''    def planCartesianMotion(self, msg, *args):
        name = args[0]
        self.move_group.set_planning_pipeline_id("ompl")
        self.move_group.set_planner_id("RRTConnect")
        print (f"planning cartesian {msg.name}")
        if DEBUG:
            print (f"   from {self._get_current_pose()}")
            print (f"   through {msg.poses}")
        rospy.sleep(DELAY)
        try:
            target_poses = msg.poses
            start_pose = self._safe_current_pose_log()
            known_scene_objects = self._moveit_known_scene_objects()
            if not target_poses:
                self._write_moveit_planning_log({
                    "request_type": "cartesian",
                    "request_topic": f"{self.name}/request/cartesian",
                    "plan_name": msg.name,
                    "planner_pipeline": "ompl",
                    "planner_id": "RRTConnect",
                    "cartesian_branch": "ompl_empty",
                    "current_pose": start_pose,
                    "target_poses": [],
                    "known_scene_objects": known_scene_objects,
                    "success": False,
                    "status": "planning failed",
                    "trajectory_points": 0,
                })
                self.planning_status_publisher.publish(String("planning failed"))
                return
            segments = []
            self.move_group.set_start_state_to_current_state()
            for segment_index, target_pose in enumerate(target_poses):
                result = self.move_group.plan(joints = target_pose)
                output = self._moveit_plan_tuple_value(result, 1)
                self._write_moveit_planning_log({
                    "request_type": "cartesian",
                    "request_topic": f"{self.name}/request/cartesian",
                    "plan_name": msg.name,
                    "planner_pipeline": "ompl",
                    "planner_id": "RRTConnect",
                    "cartesian_branch": "ompl_segment",
                    "cartesian_segment_index": segment_index,
                    "current_pose": start_pose,
                    "target_pose": self._moveit_pose_to_dict(target_pose),
                    "target_poses": self._moveit_pose_list_to_dicts(target_poses),
                    "known_scene_objects": known_scene_objects,
                    "success": bool(self._moveit_plan_tuple_value(result, 0)),
                    "planning_time": self._moveit_plan_tuple_value(result, 2),
                    "moveit_error_code": self._moveit_error_code_to_int(self._moveit_plan_tuple_value(result, 3)),
                    "trajectory_points": self._moveit_trajectory_point_count(output),
                })
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
                if DEBUG: print (f"   >> cartesian rrtconnect movement with {len(output.joint_trajectory.points)}")
                convertedTraj = output if not self.need_offset else self.joint_offset_func(output) #moveit_msgs/RobotTrajectory
                if msg.name == "hololens_path_lin" or self.store_plan:
                    with open(f'{self.root}/{name}/{msg.name}.yaml', 'w') as fp:
                        yaml.dump(output, fp, default_flow_style=True)

                self._publish_trajectory(convertedTraj.joint_trajectory, msg.name)

                if msg.name in self.trajectory_data.keys():
                    print (f"warn: overwriting previous trajectory {msg.name}")
                self.trajectory_data[msg.name] = output
                self._write_moveit_planning_log({
                    "request_type": "cartesian",
                    "request_topic": f"{self.name}/request/cartesian",
                    "plan_name": msg.name,
                    "planner_pipeline": "ompl",
                    "planner_id": "RRTConnect",
                    "cartesian_branch": "ompl_combined",
                    "cartesian_segment_index": "combined",
                    "current_pose": start_pose,
                    "target_poses": self._moveit_pose_list_to_dicts(target_poses),
                    "known_scene_objects": known_scene_objects,
                    "success": True,
                    "trajectory_points": self._moveit_trajectory_point_count(output),
                })
                self.planning_status_publisher.publish(String("success"))
            else:
                self._write_moveit_planning_log({
                    "request_type": "cartesian",
                    "request_topic": f"{self.name}/request/cartesian",
                    "plan_name": msg.name,
                    "planner_pipeline": "ompl",
                    "planner_id": "RRTConnect",
                    "cartesian_branch": "ompl_combined",
                    "cartesian_segment_index": "combined",
                    "current_pose": start_pose,
                    "target_poses": self._moveit_pose_list_to_dicts(target_poses),
                    "known_scene_objects": known_scene_objects,
                    "success": False,
                    "status": "planning failed",
                    "trajectory_points": 0,
                })
                self.planning_status_publisher.publish(String("planning failed"))

        except Exception as e:
            self._write_moveit_planning_log({
                "request_type": "cartesian",
                "request_topic": f"{self.name}/request/cartesian",
                "plan_name": getattr(msg, "name", None),
                "planner_pipeline": "ompl",
                "planner_id": "RRTConnect",
                "success": False,
                "exception": str(e),
                "traceback": traceback.format_exc(),
            })
            print(e)
        finally:
            self.move_group.set_start_state_to_current_state()

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
UR10_KINEMATICS_SETTINGS = {
    "kinematics_solver_timeout": "0.1",
    "goal_joint_tolerance": "0.005",
    "goal_position_tolerance": "0.01",
    "goal_orientation_tolerance": "0.05",
}
UR10_MOVEIT_HOME_JOINT_VALUES = {
    "shoulder_pan_joint": -0.05903655687441045,
    "shoulder_lift_joint": -1.5698241536486712,
    "elbow_joint": 1.529440704976217,
    "wrist_1_joint": -0.0015873473933716298,
    "wrist_2_joint": 1.4997673034667969,
    "wrist_3_joint": 0.0008195281261578202,
}
UR10_MOVEIT_HOME_JOINT_ORDER = (
    "elbow_joint",
    "shoulder_lift_joint",
    "shoulder_pan_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def _write_if_changed(path: Path, text: str) -> bool:
    if path.read_text() == text:
        return False
    path.write_text(text)
    return True


def _ur10_home_group_state() -> str:
    lines = ['    <group_state name="home" group="arm">']
    for joint_name in UR10_MOVEIT_HOME_JOINT_ORDER:
        lines.append(
            f'        <joint name="{joint_name}" value="{UR10_MOVEIT_HOME_JOINT_VALUES[joint_name]}"/>'
        )
    lines.append("    </group_state>")
    return "\n".join(lines)


def _with_ur10_home_group_state(srdf_text: str) -> tuple[str, bool]:
    home_state = _ur10_home_group_state()
    start_marker = '    <group_state name="home" group="arm">'
    end_marker = "    </group_state>"
    start = srdf_text.find(start_marker)
    if start != -1:
        end = srdf_text.find(end_marker, start + len(start_marker))
        if end == -1:
            raise SystemExit("Expected closing home group_state not found in UR10 SRDF")
        end += len(end_marker)
        current = srdf_text[start:end]
        if current == home_state:
            return srdf_text, False
        return f"{srdf_text[:start]}{home_state}{srdf_text[end:]}", True

    virtual_joint_marker = (
        '    <virtual_joint name="virtual_joint" type="fixed" parent_frame="world" child_link="base_link"/>'
    )
    if virtual_joint_marker not in srdf_text:
        raise SystemExit("Expected virtual joint marker not found in UR10 SRDF")
    return srdf_text.replace(virtual_joint_marker, f"{home_state}\n{virtual_joint_marker}", 1), True


def _replace_method_block(text: str, start_marker: str, end_marker: str, replacement: str, idempotence_marker: str) -> tuple[str, bool]:
    start = text.find(start_marker)
    if start == -1:
        return text, False
    end = text.find(end_marker, start + len(start_marker))
    if end == -1:
        return text, False
    del idempotence_marker
    current = text[start:end]
    if current.replace("\r\n", "\n").strip() == replacement.replace("\r\n", "\n").strip():
        return text, False
    return f"{text[:start]}{replacement}{text[end:]}", True


def _insert_planning_log_helpers(text: str) -> tuple[str, bool]:
    changed = False
    if "import json" not in text:
        text = f"{PLANNING_LOG_IMPORTS}{text}"
        changed = True
    if "def _write_moveit_planning_log(self, record):" in text:
        return text, changed
    candidates = [
        index
        for index in (
            text.find(FREE_METHOD),
            text.find(COMBINE_SAMPLED_METHOD),
            text.find(CARTESIAN_METHOD),
            text.find("    def init_topics(self):\n"),
        )
        if index != -1
    ]
    if not candidates:
        return text, changed
    insert_at = min(candidates)
    text = f"{text[:insert_at]}{PLANNING_LOG_METHODS}{text[insert_at:]}"
    return text, True


def _insert_ar_preview_scaling(text: str) -> tuple[str, bool]:
    if PUBLISH_TRAJECTORY_METHOD not in text:
        return text, False

    changed = False
    if "import copy\n" not in text:
        if "import traceback\n" in text:
            text = text.replace("import traceback\n", "import copy\nimport traceback\n", 1)
        else:
            text = f"import copy\n{text}"
        changed = True

    if AR_PREVIEW_TIME_SCALE_LINE not in text:
        if "import traceback\n" in text:
            text = text.replace("import traceback\n", f"import traceback\n\n{AR_PREVIEW_TIME_SCALE_LINE}", 1)
        elif "import copy\n" in text:
            text = text.replace("import copy\n", f"import copy\n\n{AR_PREVIEW_TIME_SCALE_LINE}", 1)
        else:
            text = f"{AR_PREVIEW_TIME_SCALE_LINE}{text}"
        changed = True

    if "def _ar_preview_trajectory(self, joint_trajectory):" not in text:
        publish_start = text.find(PUBLISH_TRAJECTORY_METHOD)
        text = f"{text[:publish_start]}{AR_PREVIEW_METHODS}{text[publish_start:]}"
        changed = True

    if RAW_PUBLISH_TRAJECTORY_ASSIGNMENT in text:
        text = text.replace(
            RAW_PUBLISH_TRAJECTORY_ASSIGNMENT,
            AR_PREVIEW_PUBLISH_TRAJECTORY_ASSIGNMENT,
            1,
        )
        changed = True
    elif AR_PREVIEW_PUBLISH_TRAJECTORY_ASSIGNMENT not in text:
        raise SystemExit(f"Expected planned trajectory assignment not found in {ROBOT_PY}")

    return text, changed


def patch_vizor_robot_py() -> bool:
    text = ROBOT_PY.read_text()
    changed = False

    for old, new in (
        (OLD_PILZ_PIPELINE, NEW_OMPL_PIPELINE),
        (OLD_PILZ_PTP_PLANNER, NEW_RRTCONNECT_PLANNER),
        (OLD_PILZ_LIN_PLANNER, NEW_RRTCONNECT_PLANNER),
    ):
        if old in text:
            text = text.replace(old, new)
            changed = True

    text, helper_changed = _insert_planning_log_helpers(text)
    changed = changed or helper_changed

    if OLD_JUMP_LINE in text:
        text = text.replace(OLD_JUMP_LINE, "", 1)
        changed = True
    if OLD_CALL in text:
        text = text.replace(OLD_JUMP_LINE, "", 1)
        text = text.replace(OLD_CALL, NEW_CALL, 1)
        changed = True
    elif "compute_cartesian_path(" in text and "avoid_collisions=True" not in text:
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

    for old_ground_plane_pose in OLD_GROUND_PLANE_POSES:
        if old_ground_plane_pose in text:
            text = text.replace(old_ground_plane_pose, NEW_GROUND_PLANE_POSE, 1)
            changed = True
            break
    else:
        if NEW_GROUND_PLANE_POSE in text:
            pass
        else:
            raise SystemExit(f"Expected ground plane pose line not found in {ROBOT_PY}")

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
        cartesian_start = text.find(CARTESIAN_METHOD)
        if cartesian_start == -1:
            raise SystemExit(f"Expected planCartesianMotion method not found in {ROBOT_PY}")
        combine_start = text.find(COMBINE_SAMPLED_METHOD)
        if combine_start != -1 and combine_start < cartesian_start:
            text = f"{text[:combine_start]}{SAMPLED_METHODS}{text[cartesian_start:]}"
        else:
            text = text.replace(CARTESIAN_METHOD, f"{SAMPLED_METHODS}{CARTESIAN_METHOD}", 1)
        changed = True
    else:
        text, sampled_changed = _replace_method_block(
            text,
            COMBINE_SAMPLED_METHOD,
            CARTESIAN_METHOD,
            SAMPLED_METHODS,
            '"request_type": "sampled"',
        )
        changed = changed or sampled_changed

    text, free_changed = _replace_method_block(
        text,
        FREE_METHOD,
        COMBINE_SAMPLED_METHOD,
        FREE_METHOD_LOGGED,
        '"request_type": "free"',
    )
    changed = changed or free_changed

    text, cartesian_changed = _replace_method_block(
        text,
        CARTESIAN_METHOD,
        PLAN_TRANSITION_METHOD,
        CARTESIAN_METHOD_LOGGED,
        '"cartesian_branch": "compute_cartesian_path"',
    )
    changed = changed or cartesian_changed

    text, ar_preview_changed = _insert_ar_preview_scaling(text)
    changed = changed or ar_preview_changed

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
    srdf_text, home_changed = _with_ur10_home_group_state(srdf_text)
    changed = changed or home_changed
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


def patch_ur10_kinematics_settings(src_root: Path = CATKIN_SRC) -> bool:
    kinematics = src_root / "moveit_support" / "ur10_moveit_support" / "config" / "kinematics.yaml"
    lines = kinematics.read_text().splitlines(keepends=True)
    patched_lines = []
    found_settings = set()

    for line in lines:
        stripped = line.lstrip()
        key = stripped.split(":", 1)[0]
        if key in UR10_KINEMATICS_SETTINGS:
            indent = line[: len(line) - len(stripped)]
            newline = "\n" if line.endswith("\n") else ""
            patched_lines.append(f"{indent}{key}: {UR10_KINEMATICS_SETTINGS[key]}{newline}")
            found_settings.add(key)
        else:
            patched_lines.append(line)

    missing = sorted(set(UR10_KINEMATICS_SETTINGS) - found_settings)
    if missing:
        raise SystemExit(f"Expected kinematics settings not found in {kinematics}: {', '.join(missing)}")

    return _write_if_changed(kinematics, "".join(patched_lines))


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
    kinematics_changed = patch_ur10_kinematics_settings()
    regenerate_ur10_with_gripper_urdf()
    print(f"Patched {ROBOT_PY}" if robot_changed else f"{ROBOT_PY} already patched")
    print("Patched Robotiq 2F-85 integration" if robotiq_changed else "Robotiq 2F-85 integration already patched")
    print("Patched MTC move_group capability" if mtc_changed else "MTC move_group capability already patched or no hook found")
    print("Patched UR10 kinematics settings" if kinematics_changed else "UR10 kinematics settings already patched")


if __name__ == "__main__":
    main()

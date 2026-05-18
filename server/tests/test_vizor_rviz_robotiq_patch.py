from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_patch_module():
    script = Path(__file__).resolve().parents[2] / "docker" / "vizor-rviz" / "patch-vizor-robot.py"
    spec = importlib.util.spec_from_file_location("patch_vizor_robot", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_mtc_module(monkeypatch):
    script = Path(__file__).resolve().parents[2] / "docker" / "vizor-rviz" / "vizor_mtc_pick_server.py"

    rospy = types.ModuleType("rospy")
    rospy.get_param = lambda *_args, **_kwargs: {}
    rospy.logerr = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "rospy", rospy)

    rospkg = types.ModuleType("rospkg")
    rospkg.ResourceNotFound = RuntimeError
    monkeypatch.setitem(sys.modules, "rospkg", rospkg)

    std_srvs = types.ModuleType("std_srvs")
    srv = types.ModuleType("std_srvs.srv")

    class Trigger:
        pass

    class TriggerResponse:
        def __init__(self, *, success=False, message=""):
            self.success = success
            self.message = message

    srv.Trigger = Trigger
    srv.TriggerResponse = TriggerResponse
    std_srvs.srv = srv
    monkeypatch.setitem(sys.modules, "std_srvs", std_srvs)
    monkeypatch.setitem(sys.modules, "std_srvs.srv", srv)

    spec = importlib.util.spec_from_file_location("vizor_mtc_pick_server_under_test", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_vizor_rviz_dockerfile_installs_robotiq_python_runtime_dependencies():
    dockerfile = Path(__file__).resolve().parents[2] / "docker" / "vizor-rviz" / "Dockerfile"

    contents = dockerfile.read_text()

    assert "python3-arrow" in contents
    assert "python3-pymodbus" in contents
    assert "python3-serial" in contents


def test_vizor_rviz_dockerfile_installs_noetic_mtc_packages():
    dockerfile = Path(__file__).resolve().parents[2] / "docker" / "vizor-rviz" / "Dockerfile"

    contents = dockerfile.read_text()

    assert "ros-noetic-moveit-task-constructor-core" in contents
    assert "ros-noetic-moveit-task-constructor-msgs" in contents
    assert "ros-noetic-moveit-task-constructor-capabilities" in contents
    assert "ros-noetic-moveit-task-constructor-visualization" in contents
    assert "ros-noetic-py-binding-tools" in contents
    assert "ros-noetic-rviz-marker-tools" in contents


def test_vizor_rviz_dockerfile_normalizes_windows_shell_script_line_endings():
    repo_root = Path(__file__).resolve().parents[2]
    dockerfile = repo_root / "docker" / "vizor-rviz" / "Dockerfile"
    attributes = repo_root / ".gitattributes"

    dockerfile_contents = dockerfile.read_text()
    attributes_contents = attributes.read_text()

    assert "sed -i 's/\\r$//'" in dockerfile_contents
    assert "/usr/local/bin/start-vizor-desktop.sh" in dockerfile_contents
    assert "*.sh text eol=lf" in attributes_contents


def test_ur10_rviz_config_loads_mtc_visualization():
    rviz_config = Path(__file__).resolve().parents[2] / "docker" / "vizor-rviz" / "ur10_robot.rviz"

    contents = rviz_config.read_text()

    assert contents.count("Class: moveit_task_constructor/Motion Planning Tasks") >= 2
    assert "Name: MTC Motion Planning Tasks" in contents
    assert "Task Solution Topic: /solution" in contents
    assert "Robot Description: robot_description" in contents
    assert "Robot Description: UR10/robot_description" in contents


def test_mtc_pick_service_reports_missing_python_api_without_fake_success(monkeypatch):
    module = _load_mtc_module(monkeypatch)
    monkeypatch.setattr(
        module,
        "_check_mtc_availability",
        lambda: {
            "package_paths": {
                "moveit_task_constructor_core": "/opt/ros/noetic/share/moveit_task_constructor_core",
                "moveit_task_constructor_msgs": "/opt/ros/noetic/share/moveit_task_constructor_msgs",
            },
            "missing_packages": [],
            "imported_modules": [],
            "missing_modules": ["moveit.task_constructor", "moveit_task_constructor"],
        },
    )

    result = module._response_payload("UR10", "beam_001", "top")

    assert result["ok"] is False
    assert result["backend"] == "mtc"
    assert result["failed_stage"] == "check_mtc_python_api"
    assert result["candidate_count"] == 0
    assert result["candidate_attempts"][0]["ok"] is False
    assert result["selected_cost"] is None
    assert result["selected_grasp_face"] == "top"
    assert "Python" in result["blocker"]
    assert "correction" in result
    assert {stage["stage_type"] for stage in result["stage_summaries"]} >= {
        "CurrentState",
        "Connect",
        "GenerateGraspPose",
        "ComputeIK",
        "MoveRelative",
        "ModifyPlanningScene",
    }
    assert result["gripper_responsibility"]["close"] == "execute_task_solution"
    assert result["attach_responsibility"]["attach"] == "mtc_modify_planning_scene"


def test_mtc_pick_service_blocks_construct_when_semantic_config_is_not_bound(monkeypatch):
    module = _load_mtc_module(monkeypatch)
    monkeypatch.setattr(
        module,
        "_check_mtc_availability",
        lambda: {
            "package_paths": {
                "moveit_task_constructor_core": "/opt/ros/noetic/share/moveit_task_constructor_core",
                "moveit_task_constructor_msgs": "/opt/ros/noetic/share/moveit_task_constructor_msgs",
            },
            "missing_packages": [],
            "imported_modules": ["moveit.task_constructor", "pymoveit_mtc.core", "pymoveit_mtc.stages"],
            "missing_modules": [],
        },
    )

    result = module._response_payload("UR10", "beam_001", None)

    assert result["ok"] is False
    assert result["backend"] == "mtc"
    assert result["failed_stage"] == "construct_pick_task"
    assert result["candidate_count"] == 0
    assert result["candidate_attempts"][0]["failed_stage"] == "construct_pick_task"
    assert result["selected_cost"] is None
    assert result["selected_grasp_face"] is None
    assert "semantic" in result["blocker"].lower()
    assert "typed" in result["correction"].lower()


def test_mtc_compound_service_parses_request_and_fails_closed(monkeypatch):
    module = _load_mtc_module(monkeypatch)
    monkeypatch.setattr(
        module,
        "_check_mtc_availability",
        lambda: {
            "package_paths": {
                "moveit_task_constructor_core": "/opt/ros/noetic/share/moveit_task_constructor_core",
                "moveit_task_constructor_msgs": "/opt/ros/noetic/share/moveit_task_constructor_msgs",
            },
            "missing_packages": [],
            "imported_modules": ["moveit.task_constructor", "pymoveit_mtc.core", "pymoveit_mtc.stages"],
            "missing_modules": [],
            "module_errors": {},
        },
    )
    request = {
        "robot_name": "UR10",
        "backend": "mtc",
        "requirements": {
            "goal": "pick_place",
            "object_name": "beam_001",
            "target_position": {"x": 0.5, "y": 0.1, "z": 0.2},
        },
        "stage_intents": ["pick", "place"],
    }

    result = module._compound_response_payload(request)

    assert result["ok"] is False
    assert result["backend"] == "mtc"
    assert result["task_kind"] == "compound"
    assert result["robot_name"] == "UR10"
    assert result["object_name"] == "beam_001"
    assert result["task_goal"] == "pick_place"
    assert result["stage_intents"] == ["pick", "place"]
    assert result["target_position"] == {"x": 0.5, "y": 0.1, "z": 0.2}
    assert result["failed_stage"] == "construct_compound_task"
    assert result["error"] == "mtc_compound_not_implemented"
    assert result["candidate_count"] == 0
    assert result["selected_cost"] is None
    assert result["task_stages"][0]["intent"] == "pick"
    assert result["task_stages"][1]["intent"] == "place"
    assert result["scene_snapshot"] == {}
    assert result["object_context"] == {}
    assert result["selected_stage_evidence"] == []
    assert result["selected_grasp_evidence"] == {}
    assert result["selected_place_evidence"] == {}
    assert result["execution_contract"]["can_execute"] is False
    assert "task_solution_id" not in result
    assert "not implemented" in result["blocker"].lower()


def test_mtc_compound_service_rejects_incomplete_request_without_solution_id(monkeypatch):
    module = _load_mtc_module(monkeypatch)

    result = module._compound_response_payload({"robot_name": "UR10", "backend": "mtc", "stage_intents": ["pick"]})

    assert result["ok"] is False
    assert result["backend"] == "mtc"
    assert result["failed_stage"] == "validate_compound_request"
    assert result["candidate_count"] == 0
    assert result["task_stages"] == []
    assert result["execution_contract"]["can_execute"] is False
    assert "task_solution_id" not in result


def test_mtc_compound_service_rejects_unsupported_goal_without_solution_id(monkeypatch):
    module = _load_mtc_module(monkeypatch)

    result = module._compound_response_payload(
        {
            "robot_name": "UR10",
            "backend": "mtc",
            "requirements": {
                "goal": "approach_hold_adjust_release",
                "object_name": "beam_001",
                "target_position": {"x": 0.5, "y": 0.1, "z": 0.2},
            },
        }
    )

    assert result["ok"] is False
    assert result["error"] == "unsupported_compound_goal"
    assert result["failed_stage"] == "validate_compound_goal"
    assert result["task_goal"] == "approach_hold_adjust_release"
    assert result["execution_contract"]["can_execute"] is False
    assert "release" in result["correction"]
    assert "task_solution_id" not in result


def test_mtc_compound_service_reports_missing_python_api_without_fake_solution(monkeypatch):
    module = _load_mtc_module(monkeypatch)
    monkeypatch.setattr(
        module,
        "_check_mtc_availability",
        lambda: {
            "package_paths": {
                "moveit_task_constructor_core": "/opt/ros/noetic/share/moveit_task_constructor_core",
                "moveit_task_constructor_msgs": "/opt/ros/noetic/share/moveit_task_constructor_msgs",
            },
            "missing_packages": [],
            "imported_modules": [],
            "missing_modules": ["moveit.task_constructor", "pymoveit_mtc.core", "pymoveit_mtc.stages"],
            "module_errors": {"moveit.task_constructor": "No module named moveit.task_constructor"},
        },
    )

    result = module._compound_response_payload(
        {
            "robot_name": "UR10",
            "backend": "mtc",
            "requirements": {"goal": "hold", "object_name": "beam_001"},
            "preferences": {"grasp_face": "top"},
        }
    )

    assert result["ok"] is False
    assert result["error"] == "mtc_python_api_unavailable"
    assert result["failed_stage"] == "check_mtc_python_api"
    assert result["requirements"] == {"goal": "hold", "object_name": "beam_001"}
    assert result["preferences"] == {"grasp_face": "top"}
    assert result["availability"]["missing_modules"] == [
        "moveit.task_constructor",
        "pymoveit_mtc.core",
        "pymoveit_mtc.stages",
    ]
    assert result["execution_contract"]["can_execute"] is False
    assert result["preview"]["solution_topic"] == "/solution"
    assert result["preview"]["solution_preview"] == "not_published"
    assert result["preview"]["ar_preview_mode"] == "unavailable"
    assert "task_solution_id" not in result


def test_mtc_compound_service_does_not_fake_plain_release_without_held_object_proof(monkeypatch):
    module = _load_mtc_module(monkeypatch)
    monkeypatch.setattr(
        module,
        "_check_mtc_availability",
        lambda: {
            "package_paths": {
                "moveit_task_constructor_core": "/opt/ros/noetic/share/moveit_task_constructor_core",
                "moveit_task_constructor_msgs": "/opt/ros/noetic/share/moveit_task_constructor_msgs",
            },
            "missing_packages": [],
            "imported_modules": ["moveit.task_constructor", "pymoveit_mtc.core", "pymoveit_mtc.stages"],
            "missing_modules": [],
            "module_errors": {},
        },
    )

    result = module._compound_response_payload(
        {
            "robot_name": "UR10",
            "backend": "mtc",
            "requirements": {"goal": "release", "object_name": "beam_001"},
        }
    )

    assert result["ok"] is False
    assert result["task_goal"] == "release"
    assert result["error"] == "mtc_compound_not_implemented"
    assert result["preview"]["ar_preview_mode"] != "none_no_motion"
    assert result["execution_contract"]["can_execute"] is False
    assert "task_solution_id" not in result


def test_patch_mtc_move_group_capability_wires_execute_task_solution(tmp_path):
    module = _load_patch_module()
    root = tmp_path / "catkin_ws" / "src"
    (root / "vizor_package" / "launch").mkdir(parents=True)
    load_robot = root / "vizor_package" / "launch" / "load_robot.launch"
    load_robot.write_text(
        """<launch>
  <arg name="robot_name" default="GoFa1"/>
  <arg name="moveit_pkg_name" default="crb15000_moveit_support"/>
  <group ns="$(arg robot_name)">
    <include file="$(find $(arg moveit_pkg_name))/launch/move_group.launch">
      <arg name="allow_trajectory_execution" value="true"/>
    </include>
  </group>
</launch>
"""
    )

    assert module.patch_mtc_move_group_capability(root) is True

    patched = load_robot.read_text()
    assert 'name="move_group_capabilities"' in patched
    assert "move_group/ExecuteTaskSolutionCapability" in patched
    assert '<arg name="capabilities" value="$(arg move_group_capabilities)"/>' in patched
    assert module.patch_mtc_move_group_capability(root) is False


def test_vizor_desktop_aliases_ur10_robot_description_for_mtc_rviz():
    start_script = Path(__file__).resolve().parents[2] / "docker" / "vizor-rviz" / "start-vizor-desktop.sh"

    contents = start_script.read_text()

    assert "/UR10/robot_description" in contents
    assert "/robot_description" in contents
    assert "/UR10/robot_description_semantic" in contents
    assert "/robot_description_semantic" in contents
    assert "/UR10/robot_description_kinematics" in contents
    assert "/robot_description_kinematics" in contents
    assert "/UR10/robot_description_planning" in contents
    assert "/robot_description_planning" in contents


def test_vizor_desktop_routes_moveit_planning_logs_to_mounted_log_dir():
    start_script = Path(__file__).resolve().parents[2] / "docker" / "vizor-rviz" / "start-vizor-desktop.sh"

    contents = start_script.read_text()

    assert "MOVEIT_PLANNING_LOG_PATH" in contents
    assert "/root/catkin_ws/logs/moveit_planning/moveit_planning.jsonl" in contents
    assert "ROS_LOG_DIR" in contents
    assert "/root/catkin_ws/logs/moveit_planning/ros" in contents
    assert "mkdir -p" in contents


def test_workshop_compose_mounts_moveit_planning_logs_into_vizor_demo():
    compose = Path(__file__).resolve().parents[2] / "docker" / "compose" / "workshop.yml"

    contents = compose.read_text()

    assert "volumes:" in contents
    assert "../../server/logs/moveit_planning:/root/catkin_ws/logs/moveit_planning" in contents


def test_patch_vizor_robot_py_uses_base_link_world_pose(monkeypatch, tmp_path):
    module = _load_patch_module()
    robot_py = tmp_path / "robot.py"
    robot_py.write_text(
        '''def create_world_pose(z = 0):
    worldXY = PoseStamped()
    header = Header()
    header.frame_id = "base" # GoFa
    # header.frame_id = "world" # Timberley/Tintin
    worldXY.header = header
    pose = Pose()
    pose.position.z = z
    worldXY.pose = pose

def plan():
    if True:
        if True:
            jump_threshold = 0.0 #0.15
            if True:
                result = self.move_group.compute_cartesian_path(target_poses, eef_step, jump_threshold) #default avoid collision

class Robot:
    def add_ground_plane(self):
        name = "ground_plane"
        ground_pose = create_world_pose(z = -0.01)
        self.scene.add_box(name, ground_pose, size=(5, 5, 0.01))

    def init_topics(self):
        rospy.Subscriber(f"{self.name}/request/cartesian", PlanningCartesian, self.planCartesianMotion, callback_args=self.name)

    def planCartesianMotion(self, msg, *args):
        pass
''',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "ROBOT_PY", robot_py)

    assert module.patch_vizor_robot_py() is True

    patched = robot_py.read_text(encoding="utf-8")
    assert 'header.frame_id = "base_link"' in patched
    assert 'header.frame_id = "base" # GoFa' not in patched
    assert "pose.orientation.w = 1.0" in patched
    assert "ground_pose = create_world_pose(z = -0.105)" in patched
    assert "ground_pose = create_world_pose(z = -0.01)" not in patched
    assert "avoid_collisions=True" in patched
    assert module.patch_vizor_robot_py() is False


def test_patch_vizor_robot_py_injects_moveit_planning_log_helpers(monkeypatch, tmp_path):
    module = _load_patch_module()
    robot_py = tmp_path / "robot.py"
    robot_py.write_text(
        '''def create_world_pose(z = 0):
    worldXY = PoseStamped()
    header = Header()
    header.frame_id = "base_link"
    worldXY.header = header
    pose = Pose()
    pose.position.z = z
    pose.orientation.w = 1.0
    worldXY.pose = pose

class Robot:
    def __init__(self):
        rospy.Subscriber(f"{self.name}/request/cartesian", PlanningCartesian, self.planCartesianMotion, callback_args=self.name)

    def add_ground_plane(self):
        name = "ground_plane"
        ground_pose = create_world_pose(z = -0.105)
        self.scene.add_box(name, ground_pose, size=(5, 5, 0.01))

    def planFreeMotion(self, msg, *args):
        pass

    def _combine_sampled_segments(self, segments):
        pass

    def planCartesianMotion(self, msg, *args):
        result = self.move_group.compute_cartesian_path(
            target_poses,
            eef_step,
            avoid_collisions=True,
        ) #default avoid collision
''',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "ROBOT_PY", robot_py)

    assert module.patch_vizor_robot_py() is True

    patched = robot_py.read_text(encoding="utf-8")
    assert "MOVEIT_PLANNING_LOG_PATH" in patched
    assert "def _write_moveit_planning_log(self, record):" in patched
    assert '"schema": "moveit_planning_diagnostics.v1"' in patched
    assert "json.dumps" in patched
    assert module.patch_vizor_robot_py() is False


def test_patch_vizor_robot_py_logs_free_planning_details(monkeypatch, tmp_path):
    module = _load_patch_module()
    robot_py = tmp_path / "robot.py"
    robot_py.write_text(
        '''def create_world_pose(z = 0):
    worldXY = PoseStamped()
    header = Header()
    header.frame_id = "base_link"
    worldXY.header = header
    pose = Pose()
    pose.position.z = z
    pose.orientation.w = 1.0
    worldXY.pose = pose

class Robot:
    def add_ground_plane(self):
        name = "ground_plane"
        ground_pose = create_world_pose(z = -0.105)
        self.scene.add_box(name, ground_pose, size=(5, 5, 0.01))

    def init_topics(self):
        rospy.Subscriber(f"{self.name}/request/cartesian", PlanningCartesian, self.planCartesianMotion, callback_args=self.name)

    def planFreeMotion(self, msg, *args):
        # rospy.wait_for_service('plan_free_motion')
        name = args[0]
        self.move_group.set_planner_id("PTP")
        print (f"planning free {msg.name}")
        if DEBUG:
            print (f"   from {self._get_current_pose()}")
            print (f"   to {msg.target_pose}")
        rospy.sleep(DELAY)
        try:
            target_pose = msg.target_pose
            result = self.move_group.plan(joints = target_pose)
            if result[0]:
                output = result[1]
                if DEBUG: print (f"   >> planning time {result[2]}")
                print (f"   >> ptp movement with {len(output.joint_trajectory.points)}")
            else:
                print(f"planning failed {result[3]}")
            code = int(str(result[3]).split(':')[-1])
            self.planning_status_publisher.publish(String(f"{self.planningResponseForHumans(code)}"))
        except Exception as e:
            print(e)

    def _combine_sampled_segments(self, segments):
        pass

    def planCartesianMotion(self, msg, *args):
        result = self.move_group.compute_cartesian_path(
            target_poses,
            eef_step,
            avoid_collisions=True,
        ) #default avoid collision
''',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "ROBOT_PY", robot_py)

    assert module.patch_vizor_robot_py() is True

    patched = robot_py.read_text(encoding="utf-8")
    assert '"request_type": "free"' in patched
    assert '"planner_pipeline": "pilz_industrial_motion_planner"' in patched
    assert '"planner_id": "PTP"' in patched
    assert '"target_pose": self._moveit_pose_to_dict(target_pose)' in patched
    assert '"planning_time": self._moveit_plan_tuple_value(result, 2)' in patched
    assert '"moveit_error_code": self._moveit_error_code_to_int(self._moveit_plan_tuple_value(result, 3))' in patched
    assert '"trajectory_points": self._moveit_trajectory_point_count(output)' in patched
    assert module.patch_vizor_robot_py() is False


def test_patch_vizor_robot_py_logs_cartesian_planning_details(monkeypatch, tmp_path):
    module = _load_patch_module()
    robot_py = tmp_path / "robot.py"
    robot_py.write_text(
        '''def create_world_pose(z = 0):
    worldXY = PoseStamped()
    header = Header()
    header.frame_id = "base_link"
    worldXY.header = header
    pose = Pose()
    pose.position.z = z
    pose.orientation.w = 1.0
    worldXY.pose = pose

class Robot:
    def add_ground_plane(self):
        name = "ground_plane"
        ground_pose = create_world_pose(z = -0.105)
        self.scene.add_box(name, ground_pose, size=(5, 5, 0.01))

    def init_topics(self):
        rospy.Subscriber(f"{self.name}/request/cartesian", PlanningCartesian, self.planCartesianMotion, callback_args=self.name)

    def planCartesianMotion(self, msg, *args):
        name = args[0]
        self.move_group.set_planner_id("LIN")
        print (f"planning cartesian {msg.name}")
        rospy.sleep(DELAY)
        try:
            eef_step = 0.05 # cartesian path interpolated at the resolution of 5cm
            target_poses = msg.poses
            if len(target_poses) == 2: # use linear planner on a single pose
                result = self.move_group.plan(joints = target_poses[1])
                output = result[1]
                print(f"    >> lin movement with {len(output.joint_trajectory.points)} poses")
            else:
                result = self.move_group.compute_cartesian_path(
                    target_poses,
                    eef_step,
                    avoid_collisions=True,
                ) #default avoid collision
                output = result[0]
                print (f"   >> fraction {result[1]}")
            if output:
                self._publish_trajectory(output.joint_trajectory, msg.name)
            else:
                print(f"planning failed {result[1]}")
        except Exception as e:
            print(e)

    def plan_transition(self, joint_trajectory_point, name = "transition"):
        pass
''',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "ROBOT_PY", robot_py)

    assert module.patch_vizor_robot_py() is True

    patched = robot_py.read_text(encoding="utf-8")
    assert '"request_type": "cartesian"' in patched
    assert '"planner_id": "LIN"' in patched
    assert '"cartesian_branch": "lin_two_pose"' in patched
    assert '"cartesian_branch": "compute_cartesian_path"' in patched
    assert '"eef_step": eef_step' in patched
    assert '"jump_threshold": None' in patched
    assert '"avoid_collisions": True' in patched
    assert '"fraction": fraction' in patched
    assert '"target_poses": self._moveit_pose_list_to_dicts(target_poses)' in patched
    assert module.patch_vizor_robot_py() is False


def test_patch_vizor_robot_py_updates_previous_ground_plane_patch(monkeypatch, tmp_path):
    module = _load_patch_module()
    robot_py = tmp_path / "robot.py"
    robot_py.write_text(
        '''def create_world_pose(z = 0):
    worldXY = PoseStamped()
    header = Header()
    header.frame_id = "base_link"
    worldXY.header = header
    pose = Pose()
    pose.position.z = z
    pose.orientation.w = 1.0
    worldXY.pose = pose

def plan():
    if True:
        if True:
            eef_step = 0.05 # cartesian path interpolated at the resolution of 5cm
            if True:
                result = self.move_group.compute_cartesian_path(
                    target_poses,
                    eef_step,
                    avoid_collisions=True,
                ) #default avoid collision

class Robot:
    def add_ground_plane(self):
        name = "ground_plane"
        ground_pose = create_world_pose(z = -0.095)
        self.scene.add_box(name, ground_pose, size=(5, 5, 0.01))

    def init_topics(self):
        rospy.Subscriber(f"{self.name}/request/cartesian", PlanningCartesian, self.planCartesianMotion, callback_args=self.name)

    def planCartesianMotion(self, msg, *args):
        pass
''',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "ROBOT_PY", robot_py)

    assert module.patch_vizor_robot_py() is True

    patched = robot_py.read_text(encoding="utf-8")
    assert "ground_pose = create_world_pose(z = -0.105)" in patched
    assert "ground_pose = create_world_pose(z = -0.095)" not in patched
    assert "avoid_collisions=True" in patched
    assert module.patch_vizor_robot_py() is False


def test_patch_vizor_robot_py_wires_sampled_request_to_rrtconnect(monkeypatch, tmp_path):
    module = _load_patch_module()
    robot_py = tmp_path / "robot.py"
    robot_py.write_text(
        '''DELAY = 0

class String:
    pass

def create_world_pose(z = 0):
    worldXY = PoseStamped()
    header = Header()
    header.frame_id = "base_link"
    worldXY.header = header
    pose = Pose()
    pose.position.z = z
    pose.orientation.w = 1.0
    worldXY.pose = pose

class Robot:
    def add_ground_plane(self):
        name = "ground_plane"
        ground_pose = create_world_pose(z = -0.01)
        self.scene.add_box(name, ground_pose, size=(5, 5, 0.01))

    def __init__(self):
        rospy.Subscriber(f"{self.name}/request/free", PlanningFree, self.planFreeMotion, callback_args=self.name)
        rospy.Subscriber(f"{self.name}/request/cartesian", PlanningCartesian, self.planCartesianMotion, callback_args=self.name)

    def planCartesianMotion(self, msg, *args):
        result = self.move_group.compute_cartesian_path(
            target_poses,
            eef_step,
            avoid_collisions=True,
        ) #default avoid collision
''',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "ROBOT_PY", robot_py)

    assert module.patch_vizor_robot_py() is True

    patched = robot_py.read_text(encoding="utf-8")
    assert 'rospy.Subscriber(f"{self.name}/request/sampled", PlanningCartesian, self.planSampledApproachMotion, callback_args=self.name)' in patched
    assert "def planSampledApproachMotion(self, msg, *args):" in patched
    assert 'self.move_group.set_planning_pipeline_id("ompl")' in patched
    assert 'self.move_group.set_planner_id("RRTConnect")' in patched
    assert '"request_type": "sampled"' in patched
    assert '"planner_pipeline": "ompl"' in patched
    assert '"planner_id": "RRTConnect"' in patched
    assert '"planning_time": self._moveit_plan_tuple_value(result, 2)' in patched
    assert '"moveit_error_code": self._moveit_error_code_to_int(self._moveit_plan_tuple_value(result, 3))' in patched
    assert '"sampled_segment_index": segment_index' in patched
    assert patched.index("def planSampledApproachMotion") < patched.index("def planCartesianMotion")
    assert module.patch_vizor_robot_py() is False


def test_patch_vizor_robot_py_wires_agent_path_execute_and_stop(monkeypatch, tmp_path):
    module = _load_patch_module()
    robot_py = tmp_path / "robot.py"
    robot_py.write_text(
        '''DELAY = 0

class String:
    pass

def create_world_pose(z = 0):
    worldXY = PoseStamped()
    header = Header()
    header.frame_id = "base_link"
    worldXY.header = header
    pose = Pose()
    pose.position.z = z
    pose.orientation.w = 1.0
    worldXY.pose = pose

class Robot:
    def add_ground_plane(self):
        name = "ground_plane"
        ground_pose = create_world_pose(z = -0.01)
        self.scene.add_box(name, ground_pose, size=(5, 5, 0.01))

    def __init__(self):
        self.trajectory_data = {}
        rospy.Subscriber(f"{self.name}/request/free", PlanningFree, self.planFreeMotion, callback_args=self.name)
        rospy.Subscriber(f"{self.name}/request/cartesian", PlanningCartesian, self.planCartesianMotion, callback_args=self.name)
        rospy.Subscriber(f"{self.name}/command/execute", String, self.executeStoredMotion, callback_args=self.name)

    def executeStoredMotion(self, msg, *args):
        output = self.trajectory_data[msg.data]
        self.move_group.execute(output, wait=False)

    def planCartesianMotion(self, msg, *args):
        result = self.move_group.compute_cartesian_path(
            target_poses,
            eef_step,
            avoid_collisions=True,
        ) #default avoid collision
''',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "ROBOT_PY", robot_py)

    assert module.patch_vizor_robot_py() is True

    patched = robot_py.read_text(encoding="utf-8")
    assert 'rospy.Subscriber(f"{self.name}/command/stop", String, self.stopAgentPath, callback_args=self.name)' in patched
    assert "def executeAgentPath(self, name):" in patched
    assert "if msg.data == \"AgentPath\":" in patched
    assert "self.executeAgentPath(msg.data)" in patched
    assert "def stopAgentPath(self, msg, *args):" in patched
    assert "refusing loose stage execution" in patched
    assert "self.move_group.execute(output, wait=True)" not in patched
    assert 'self.active_agent_path = None' in patched
    assert module.patch_vizor_robot_py() is False


def test_patch_robotiq_2f85_integration_replaces_fixed_mesh_and_wires_action_server(tmp_path):
    module = _load_patch_module()
    root = tmp_path / "catkin_ws" / "src"
    (root / "urdf_support" / "ur10_support" / "urdf").mkdir(parents=True)
    (root / "moveit_support" / "ur10_moveit_support" / "config").mkdir(parents=True)
    (root / "vizor_package" / "launch").mkdir(parents=True)
    (root / "robotiq_2f_gripper_control" / "launch").mkdir(parents=True)

    (root / "urdf_support" / "ur10_support" / "urdf" / "ur10_with_gripper.xacro").write_text(
        """<?xml version="1.0"?>
<robot name="ur10_with_gripper" xmlns:xacro="http://wiki.ros.org/xacro">
  <xacro:include filename="$(find ur10_support)/urdf/ur10_macro.xacro"/>
  <xacro:include filename="$(find ur10_support)/urdf/end_effector.xacro"/>
  <xacro:ur10_robot prefix=""/>
  <xacro:gripper_df2025 parent="tool0" prefix="">
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </xacro:gripper_df2025>
</robot>
"""
    )
    (root / "moveit_support" / "ur10_moveit_support" / "config" / "ur10_with_gripper.srdf").write_text(
        """<robot name="ur10_with_gripper">
    <group name="arm">
        <chain base_link="base_link" tip_link="tcp"/>
    </group>
    <virtual_joint name="virtual_joint" type="fixed" parent_frame="world" child_link="base_link"/>
    <disable_collisions link1="gripper_base" link2="gripper_fingers" reason="Adjacent"/>
    <disable_collisions link1="gripper_base" link2="wrist_3_link" reason="Adjacent"/>
    <disable_collisions link1="gripper_fingers" link2="wrist_3_link" reason="Never"/>
</robot>
"""
    )
    (root / "vizor_package" / "launch" / "load_robot.launch").write_text(
        """<launch>
  <arg name="robot_name" default="GoFa1"/>
  <group ns="$(arg robot_name)">
        <node name="joint_state_publisher" pkg="joint_state_publisher" type="joint_state_publisher" unless="$(arg use_gui)">
            <rosparam param="source_list">[move_group/fake_controller_joint_states]</rosparam>
        </node>
  </group>
</launch>
"""
    )
    (root / "vizor_package" / "launch" / "vizor2ros.launch").write_text(
        """<launch>
  <arg name="use_UR10" default="true"/>
  <include file="$(find vizor_package)/launch/load_robot.launch" if="$(arg use_UR10)">
      <arg name="robot_name" value="UR10"/>
      <arg name="moveit_pkg_name" default="ur10_moveit_support"/>
      <arg name="use_rviz" value="$(arg use_rviz)" />
  </include>
</launch>
"""
    )
    (root / "robotiq_2f_gripper_control" / "launch" / "robotiq_action_server.launch").write_text(
        """<launch>
    <node pkg="robotiq_2f_gripper_control" type="robotiq_2f_action_server.py" name="robotiq_2f_action_server">
        <param name="joint_name" value="$(arg joint_name)" />
    </node>
</launch>
"""
    )

    module.patch_robotiq_2f85_integration(root)

    xacro = (root / "urdf_support" / "ur10_support" / "urdf" / "ur10_with_gripper.xacro").read_text()
    assert "robotiq_arg2f_85_model_macro.xacro" in xacro
    assert "end_effector.xacro" not in xacro
    assert '<child link="robotiq_arg2f_base_link"/>' in xacro
    assert '<link name="tcp"/>' in xacro

    srdf = (root / "moveit_support" / "ur10_moveit_support" / "config" / "ur10_with_gripper.srdf").read_text()
    assert "gripper_base" not in srdf
    assert "robotiq_arg2f_base_link" in srdf
    assert "left_inner_finger_pad" in srdf
    assert '<disable_collisions link1="left_inner_knuckle" link2="left_inner_finger" reason="Never"/>' in srdf
    assert '<disable_collisions link1="right_inner_knuckle" link2="right_inner_finger" reason="Never"/>' in srdf

    load_robot = (root / "vizor_package" / "launch" / "load_robot.launch").read_text()
    assert "use_robotiq_2f85" in load_robot
    assert "robotiq_action_server.launch" in load_robot
    assert "gripper_joint_states" in load_robot

    vizor2ros = (root / "vizor_package" / "launch" / "vizor2ros.launch").read_text()
    assert '<arg name="use_robotiq_2f85" default="true"/>' in vizor2ros
    assert '<arg name="use_robotiq_2f85" value="$(arg use_robotiq_2f85)"/>' in vizor2ros

    robotiq_launch = (root / "robotiq_2f_gripper_control" / "launch" / "robotiq_action_server.launch").read_text()
    assert '<remap from="/joint_states" to="gripper_joint_states"/>' in robotiq_launch

    module.patch_robotiq_2f85_integration(root)

    assert (root / "moveit_support" / "ur10_moveit_support" / "config" / "ur10_with_gripper.srdf").read_text() == srdf


def test_patch_ur10_kinematics_timeout_sets_more_reliable_ik_window(tmp_path):
    module = _load_patch_module()
    root = tmp_path / "catkin_ws" / "src"
    config = root / "moveit_support" / "ur10_moveit_support" / "config"
    config.mkdir(parents=True)
    kinematics = config / "kinematics.yaml"
    kinematics.write_text(
        """arm:
  kinematics_solver: trac_ik_kinematics_plugin/TRAC_IKKinematicsPlugin
  kinematics_solver_timeout: 0.005
  kinematics_solver_search_resolution: 0.005
"""
    )

    assert module.patch_ur10_kinematics_timeout(root) is True

    patched = kinematics.read_text()
    assert "kinematics_solver_timeout: 0.05" in patched
    assert "kinematics_solver_search_resolution: 0.005" in patched
    assert module.patch_ur10_kinematics_timeout(root) is False
    assert kinematics.read_text() == patched

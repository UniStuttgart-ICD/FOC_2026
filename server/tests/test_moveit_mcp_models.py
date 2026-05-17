from moveit_mcp.models import Evidence, ToolResult, VerificationCheck


def test_tool_result_requires_feedback_verification_and_evidence():
    result = ToolResult.pass_result(
        robot="UR10",
        tool="plan_free_motion",
        phase="planned",
        status="success! ",
        message="Plan succeeded",
        checks=[VerificationCheck(name="status_success", passed=True, details="success! ")],
        evidence=[Evidence(kind="ros_topic", topic="/UR10/request/status", summary="success! ")],
        raw={"trajectory_points": 1},
    )

    payload = result.to_dict()

    assert payload["ok"] is True
    assert payload["robot"] == "UR10"
    assert payload["tool"] == "plan_free_motion"
    assert payload["feedback"]["phase"] == "planned"
    assert payload["feedback"]["status"] == "success! "
    assert payload["feedback"]["can_execute"] is True
    assert "correction" not in payload["feedback"]
    assert payload["verification"]["result"] == "pass"
    assert payload["verification"]["checks"][0]["name"] == "status_success"
    assert payload["evidence"][0]["topic"] == "/UR10/request/status"


def test_fail_result_never_claims_can_execute():
    result = ToolResult.fail_result(
        robot="UR10",
        tool="plan_cartesian_motion",
        phase="planned",
        status="incomplete path",
        message="Cartesian path incomplete",
        correction="Retry with a smaller or safer target and execute only a successful returned raw.plan_name.",
        checks=[VerificationCheck(name="complete_cartesian_path", passed=False, details="incomplete path")],
        evidence=[],
        raw={},
    )

    payload = result.to_dict()

    assert payload["ok"] is False
    assert payload["feedback"]["can_execute"] is False
    assert "smaller or safer target" in payload["feedback"]["correction"]
    assert payload["verification"]["result"] == "fail"

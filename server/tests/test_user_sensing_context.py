import json

from user_sensing.context import UserSensingContextStore


def test_empty_user_sensing_context_renders_advisory_block() -> None:
    store = UserSensingContextStore()

    text = store.render_instruction_block()

    assert "User sensing context" in text
    assert "advisory only" in text
    assert "No user sensing has been observed yet" in text


def test_user_sensing_context_updates_from_vizor_sensor_context_output() -> None:
    store = UserSensingContextStore(time_fn=lambda: 10.0)
    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "gaze": {
                    "available": True,
                    "target": "beam_001",
                    "age_s": 0.2,
                    "stale": False,
                },
                "attention": {
                    "available": True,
                    "fresh": True,
                    "dominant_target": "beam_001",
                    "last_stable_target": "beam_001",
                    "ranked_targets": [
                        {
                            "target": "beam_001",
                            "confidence": "high",
                            "dwell_s": 3.4,
                            "last_seen_age_s": 0.2,
                        }
                    ],
                },
                "user": {
                    "available": True,
                    "position": {"x": 0.34, "y": -0.72, "z": 1.25},
                    "frame": "base_link",
                    "age_s": 0.3,
                    "stale": False,
                },
                "manual_target": {
                    "available": False,
                    "position": None,
                    "age_s": None,
                    "stale": True,
                },
            }
        }
    )

    store.update_from_tool_result(output)

    text = store.render_instruction_block()
    assert "attention target: beam_001 (high confidence, dwell 3.4s)" in text
    assert "gaze target: beam_001" in text
    assert "user position: x=0.340, y=-0.720, z=1.250, frame=base_link" in text
    assert "manual target: unavailable" in text
    assert "status age: 0.0s" in text
    assert store.summary_attributes() == {
        "context.available": True,
        "attention.available": True,
        "attention.fresh": True,
        "attention.dominant_target": "beam_001",
        "attention.last_stable_target": "beam_001",
        "gaze.available": True,
        "gaze.stale": False,
        "gaze.age_s": 0.2,
        "gaze.target": "beam_001",
        "user.available": True,
        "user.stale": False,
        "user.age_s": 0.3,
        "manual_target.available": False,
        "manual_target.stale": True,
    }
    assert "attention=beam_001" in store.summary_text()


def test_user_sensing_context_marks_stale_fields() -> None:
    store = UserSensingContextStore(time_fn=lambda: 20.0)
    store.update_from_tool_result(
        json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "gaze": {
                        "available": True,
                        "target": "beam_001",
                        "age_s": 5.2,
                        "stale": True,
                    },
                    "user": {"available": False, "position": None, "stale": True},
                    "manual_target": {"available": False, "position": None, "stale": True},
                }
            }
        )
    )

    text = store.render_instruction_block()

    assert "gaze target: beam_001 (stale, age 5.2s)" in text
    assert "user position: unavailable" in text


def test_user_sensing_context_renders_raw_gaze_object_candidate() -> None:
    store = UserSensingContextStore(time_fn=lambda: 30.0)
    store.update_from_tool_result(
        json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "gaze": {
                        "available": True,
                        "target": "5",
                        "raw_target": "dynamic_5",
                        "age_s": 0.2,
                        "stale": False,
                    },
                    "user": {"available": False, "position": None, "stale": True},
                    "manual_target": {"available": False, "position": None, "stale": True},
                }
            }
        )
    )

    text = store.render_instruction_block()

    assert "gaze target: 5" in text
    assert "gaze object candidate: dynamic_5" in text
    assert store.summary_attributes()["gaze.raw_target"] == "dynamic_5"


def test_user_sensing_context_derives_canonical_dynamic_candidate_from_numeric_gaze() -> None:
    store = UserSensingContextStore(time_fn=lambda: 31.0)
    store.update_from_tool_result(
        json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "gaze": {
                        "available": True,
                        "target": "5",
                        "age_s": 0.2,
                        "stale": False,
                    },
                    "user": {"available": False, "position": None, "stale": True},
                    "manual_target": {"available": False, "position": None, "stale": True},
                }
            }
        )
    )

    text = store.render_instruction_block()

    assert "gaze object candidate: dynamic_5" in text
    assert store.summary_attributes()["gaze.object_candidate"] == "dynamic_5"

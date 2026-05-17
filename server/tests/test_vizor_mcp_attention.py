from vizor_mcp.attention import GazeAttentionTracker


def test_attention_ranks_targets_by_recent_dwell_and_filters_flicker() -> None:
    tracker = GazeAttentionTracker(window_s=8.0, flicker_s=0.25, stable_s=0.5)

    tracker.record("beam_12", at_s=0.0)
    tracker.record("shoulder_0", at_s=1.0)
    tracker.record("beam_12", at_s=1.1)
    tracker.record("shoulder_0", at_s=1.2)
    tracker.record("shoulder_0", at_s=4.0)

    summary = tracker.summarize(now_s=5.0)

    assert summary["dominant_target"] == "shoulder_0"
    assert summary["last_stable_target"] == "shoulder_0"
    assert summary["ranked_targets"][0]["target"] == "shoulder_0"
    assert summary["ranked_targets"][0]["dwell_s"] > 3.0
    assert summary["ranked_targets"][0]["confidence"] == "high"
    assert all(
        item["target"] != "beam_12" or item["dwell_s"] >= 0.25
        for item in summary["ranked_targets"]
    )


def test_attention_decays_when_latest_gaze_is_old() -> None:
    tracker = GazeAttentionTracker(window_s=8.0, stale_after_s=2.0)

    tracker.record("column_a", at_s=0.0)

    summary = tracker.summarize(now_s=5.0)

    assert summary["dominant_target"] == "column_a"
    assert summary["fresh"] is False
    assert summary["ranked_targets"][0]["confidence"] == "low"
    assert summary["ranked_targets"][0]["last_seen_age_s"] == 5.0

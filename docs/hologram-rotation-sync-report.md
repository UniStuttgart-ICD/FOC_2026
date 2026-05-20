# Hologram Rotation And ModelTracker Sync Report

Date: 2026-05-20

## Summary

We are aligning the hologram shared-geometry model with the physical model, then rotating the hologram frame 90 degrees clockwise in the horizontal XY plane around world origin `0, 0, 0`.

The work affects two related areas:

- The static hologram geometry JSON files.
- The live ModelTracker sync path that writes new hologram poses during runtime.

The physical model remains the source of truth for the base layout. The hologram model is derived from it, then rotated.

## Repositories Touched

The work was first applied in:

```text
C:\Users\Samuel\Documents\github\pipecat\pipecat-agent
```

It was then applied to:

```text
C:\Users\Samuel\Documents\github\-FOC_2026
```

This report is saved in the `-FOC_2026` repo because that was the last requested target repo.

## Files In Scope

Static geometry:

```text
server/robot_control/shared_geometry/physical_model.json
server/robot_control/shared_geometry/physical_model - Copy.json
server/robot_control/shared_geometry/hologram_model.json
server/robot_control/shared_geometry/hologram_model - Copy.json
```

Runtime sync:

```text
server/robot_control/shared_geometry/modeltracker_sync.py
server/robot_control/shared_geometry/modeltracker_sync_server.py
```

Tests:

```text
server/tests/test_geometry_world_context.py
server/tests/test_modeltracker_sync.py
server/tests/test_modeltracker_sync_server.py
```

## Intended Geometry Rule

The physical model is not rotated by this change.

The hologram model is rebuilt from the physical model, then rotated clockwise around world origin in the XY plane.

Clockwise is interpreted as looking from above, down toward the XY plane.

The coordinate transform is:

```text
(x, y, z) -> (y, -x, z)
```

For the hologram base derived from the physical model, the resulting target positions are:

```text
dynamic_0: [0.9, -0.35,  -0.0775]
dynamic_1: [0.9, -0.264, -0.0775]
dynamic_2: [0.9, -0.178, -0.0775]
```

The hologram quaternion for these horizontally rotated base elements is:

```text
[0.0, 0.0, 1.0, 0.0]
```

## Section Size

The hologram model keeps the enlarged hologram section dimensions:

```text
y = 0.04 m
z = 0.045 m
```

That means the hologram beams use a 4 cm width and a 4.5 cm height.

The physical model dimensions are not changed by this rotation work.

## Static File Sync Rule

The two hologram files should match when they represent the static target layout:

```text
hologram_model.json
hologram_model - Copy.json
```

The copy file is used as the preserved static layout copy.

During our verified static update, both hologram files matched and both held the rotated layout.

## Runtime ModelTracker Sync Rule

ModelTracker sends live object poses into:

```text
server/robot_control/shared_geometry/modeltracker_sync.py
```

Because the hologram frame was rotated, the ModelTracker correction also had to change.

In `-FOC_2026`, the sync code uses separate position and rotation corrections.

The updated constants are:

```python
DEFAULT_POSITION_CORRECTION_RADIANS = 3.0 * math.pi / 2.0
DEFAULT_ROTATION_CORRECTION_RADIANS = 3.0 * math.pi / 2.0
```

Meaning:

- Position correction rotates incoming ModelTracker positions clockwise by 90 degrees.
- Rotation correction maps incoming ModelTracker local orientation into the hologram base orientation.
- Runtime orientation composes as `position_frame_correction * incoming_orientation * rotation_correction`.

Do not collapse those two rotations into one post-multiplied `math.pi` correction. That only works for identity or Z-only incoming orientations. For non-commuting rotations, it flips horizontal XY-plane rotations into vertical poses and vertical rotations into horizontal poses.

Example:

```text
incoming ModelTracker center: [0.2, 0.3, 0.4]
corrected hologram center:   [0.3, -0.2, 0.4]
```

## Tests Updated

The ModelTracker tests were updated so they expect the rotated frame.

Examples:

```text
old expected center: [0.2, 0.3, 0.4]
new expected center: [0.3, -0.2, 0.4]
```

The server wrapper test was updated the same way, so the HTTP sync path is also checked.

Follow-up regression tests now check that:

- A Rhino XY-plane rotation stays horizontal after sync.
- A Rhino vertical rotation stays vertical after sync.

## Verification Already Run

In `C:\Users\Samuel\Documents\github\pipecat\pipecat-agent`:

```text
uv run pytest tests/test_geometry_world_context.py tests/test_modeltracker_sync.py tests/test_modeltracker_sync_server.py
```

Result:

```text
17 passed
```

In `C:\Users\Samuel\Documents\github\-FOC_2026`:

```text
uv run pytest tests/test_geometry_world_context.py tests/test_modeltracker_sync.py tests/test_modeltracker_sync_server.py
```

Result:

```text
19 passed
```

The geometry checks also confirmed the intended rotated positions for `dynamic_0`, `dynamic_1`, and `dynamic_2`.

`git diff --check` also passed for the changed files.

## Current Important Observation

While preparing this report, `-FOC_2026` showed a new difference between:

```text
server/robot_control/shared_geometry/hologram_model.json
server/robot_control/shared_geometry/hologram_model - Copy.json
```

The copy file still held the rotated static layout:

```text
dynamic_0: [0.9, -0.35,  -0.0775]
dynamic_1: [0.9, -0.264, -0.0775]
dynamic_2: [0.9, -0.178, -0.0775]
```

But `hologram_model.json` had live-looking updates for at least `dynamic_0` and `dynamic_2`:

```text
dynamic_0: [0.742761224508, -0.041571587324, 0.140499997884]
dynamic_2: [0.733918756247, 0.265640258789, 0.350000083447]
```

That suggests a live ModelTracker process or another writer may have updated `hologram_model.json` after the verified static rotation.

This is not necessarily wrong if `hologram_model.json` is the live target file. It does mean `hologram_model - Copy.json` and `hologram_model.json` should not be assumed to stay identical while live sync is running.

## Practical Meaning

Use `physical_model.json` as the base physical layout.

Use `hologram_model - Copy.json` as the static rotated hologram layout reference.

Use `hologram_model.json` as the file that can be changed by live ModelTracker sync.

If we need both hologram files to remain identical, the live sync process must either be stopped or changed to write both files. Right now the live sync path writes the main hologram model file.

## Remaining Decision

We need to decide what role each hologram file should have:

1. `hologram_model.json` is live state, and the copy is a static reference.
2. Both hologram files must stay identical, and live sync should update both.
3. Live sync should write to a separate runtime state file, leaving both static hologram files unchanged.

Option 1 matches the current behavior best.

Option 2 is simple but makes the copy less useful as a stable reference.

Option 3 is cleaner long term, but it is a broader design change.

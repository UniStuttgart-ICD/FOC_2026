# Hybrid Manipulation Stage Planning

The staged MoveIt manipulation backend uses `free_motion` for far approach to pick pre-grasp, far approach to place pre-pose, and held-object travel, while keeping Cartesian planning for contact-sensitive final approach, lift, descent, and extraction. This keeps planning latency lower than all-Cartesian staged planning while preserving straight, predictable motion near contact; `sampled_motion` stays out of the first optimization until it is a complete task-stage planner.

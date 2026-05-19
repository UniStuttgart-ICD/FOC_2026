# Hybrid Manipulation Stage Planning

Status: superseded by the active OMPL/RRTConnect route migration.

Active `free`, `cartesian`, and `sampled` MoveIt routes now use OMPL/RRTConnect. The `cartesian` route and tool name is legacy API vocabulary; it no longer guarantees straight TCP motion.

This ADR no longer assigns local stages to Pilz LIN or `compute_cartesian_path`. Treat those as historical or installed-planner context only, not active-route behavior.

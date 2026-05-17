# Docs

This directory mixes live project references, design notes, research material, and historical plans. Treat the files below by lifecycle.

## System Of Record

- [`../ARCHITECTURE.md`](../ARCHITECTURE.md): target module boundaries and architecture invariants.
- [`../CONTEXT.md`](../CONTEXT.md): project glossary and domain language.
- [`adr/`](adr/): architecture decisions that should be hard to reverse or easy to forget.

## Operations

- [`testing.md`](testing.md): deterministic test commands.
- [`benchmarking.md`](benchmarking.md): current voice timing run notes for the bundled runtime profile.
- [`operator-dashboard.md`](operator-dashboard.md): how to launch the MAVE operator dashboard.

## Design Notes

- [`design-notes/`](design-notes/): focused HTML reports and proposal companions.
  - [`ground-plane-safety-fix.html`](design-notes/ground-plane-safety-fix.html)
  - [`robot-tooling-foundations.html`](design-notes/robot-tooling-foundations.html)
  - [`task-ledger-blackboard.html`](design-notes/task-ledger-blackboard.html)

## Research And Workshop Material

- [`research/`](research/): research notes and visual explainers.
- [`ideas/`](ideas/): workshop scenarios, card drafts, and concept exploration.
- [`presentations/`](presentations/): generated slide decks and presentation artifacts.

## Historical Plans And Specs

- [`plans/`](plans/): early implementation plans kept for provenance.
- [`superpowers/specs/`](superpowers/specs/): approved design specs from earlier workstreams.
- [`superpowers/plans/`](superpowers/plans/): detailed implementation plans from earlier workstreams.

Historical files may mention old providers, profile matrices, or stale robot tool names. Use `ARCHITECTURE.md`, `CONTEXT.md`, and `server/runtime_profiles.toml` for current behavior.

Agent process notes are local-only and ignored for workshop handover.

# Domain Docs

How engineering skills should consume this repo's domain documentation.

## Layout

This is a single-context repo.

- Domain glossary: `CONTEXT.md`
- Target architecture: `ARCHITECTURE.md`
- Architecture Decision Records: `docs/adr/`

`docs/adr/` may be empty. Create ADRs only when a decision is hard to reverse, surprising without context, and the result of a real trade-off.

## Before exploring

Read these files when relevant to the task:

1. `CONTEXT.md` for project language.
2. `ARCHITECTURE.md` for target module seams and import direction.
3. Any ADR in `docs/adr/` that touches the area being changed.

If an ADR directory or matching ADR does not exist, proceed silently.

## Use the glossary's vocabulary

When output names a domain concept, use the terms in `CONTEXT.md`. Do not drift to stale terms such as "Robot Safety" when the glossary says **Robot Call Validation** or **MoveIt Safety Boundary**.

## Flag ADR conflicts

If a recommendation contradicts an ADR, surface it explicitly rather than silently overriding it:

> _Contradicts ADR-0007 — but worth reopening because ..._

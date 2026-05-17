# AGENTS.md

Documentation workspace for the Pipecat voice robot agent.

## Project map

- `README.md` - Documentation index.
- `architecture-explorer.html` - Interactive architecture explainer.
- `adr/` - Architecture decision records.
- `agents/` - Agent workflow notes.
- `design-notes/` - Design sketches and exploratory notes.
- `ideas/` - Early proposals and rough concepts.
- `plans/` - Implementation plans.
- `presentations/` - Presentation material.
- `research/` - Research notes and findings.
- `superpowers/` - Superpowers specs and plans.

<important if="you are creating or revising HTML documentation">
- Treat [The unreasonable effectiveness of HTML](https://thariqs.github.io/html-effectiveness/) as the reference example for useful HTML docs.
- Prefer self-contained `.html` artifacts when the work benefits from layout, comparison, interaction, diagrams, timelines, cards, tabs, or visual hierarchy.
- Write the page to explain the user's goals, intentions, tradeoffs, and current choices as clearly as possible.
- Replace long linear prose with structured sections the user can scan, compare, and revisit.
- Keep the HTML doc directly useful in a browser without a build step unless the surrounding docs already require one.
</important>

<important if="an HTML doc explains architecture, flow, dependencies, decisions, or other complex concepts">
- Use visuals as helping cues: diagrams, flowcharts, timelines, callouts, comparison tables, annotated mockups, or small charts.
- Use visuals to clarify the concept, not to decorate the page.
- When precise arrows, lanes, labels, or SVG layout matter, follow `the drawing-precise-html-svg-diagrams skill`.
- For inline SVG diagrams, keep node placement, arrow routing, labels, and mobile behavior deliberate.
- Verify visual docs in a browser when practical before claiming they are ready.
</important>

<important if="you are writing explanatory documentation">
- Be succinct and direct.
- Name assumptions and tradeoffs plainly.
- Favor concrete examples, domain terms, and user-facing implications over generic explanation.
</important>

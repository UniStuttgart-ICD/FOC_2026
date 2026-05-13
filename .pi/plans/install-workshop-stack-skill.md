# install-workshop-stack skill plan

1. Use `skill-creator` and `superpowers:writing-skills`.
2. Inspect the operator dashboard entrypoint, config, and service dependencies.
3. Create RED pressure scenarios for installer-skill failure modes.
4. Initialize `install-workshop-stack` under `C:\Users\Samuel\.agents\skills`.
5. Add a concise `SKILL.md`, a reusable setup script, and one focused reference file.
6. Validate the setup script help/dry run and run the skill validator.

## RED scenarios

- Agent hardcodes Samuel-specific paths from `operator_dashboard.example.toml` and never discovers the user's local workspace.
- Agent runs the dashboard before `uv sync`, Docker, and service cwd checks, then reports a generic startup failure.
- Agent edits `.env` or commits secrets while trying to "install everything".
- Agent assumes the Vizor Docker stack and Pipecat voice agent are inside the MAVE repo.
- Agent starts long-lived services as validation instead of doing a bounded install/dry-run check.

# Testing

## Default tests

Run deterministic tests from `server/`:

```bash
uv run pytest
```

Default tests must not require live provider API keys, MoveIt MCP, STT/TTS providers, wake-word models beyond existing unit-test fixtures, browser audio, or robot simulation infrastructure.

## Exploratory gesture evals

Prompts such as `wave to me` and `draw a star` are exploratory evals. They are useful for behavior review, but they are not part of the pass/fail testing pipeline until their assertions become deterministic and actionable.

Historical live smoke and model benchmarking code is archived under `archive/model-benchmarking/`.

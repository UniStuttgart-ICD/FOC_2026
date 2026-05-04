Status: DONE
Commit: fix: harden wake word runtime integration

Validation:
- Red test run before implementation: `cd server && uv run pytest tests/test_wake_gate.py tests/test_transcript_cleanup.py -v` failed during collection because `OpenWakeWordResourceError` and `WakePhraseTranscriptCleaner` did not exist.
- `cd server && uv run pytest tests/test_wake_gate.py tests/test_transcript_cleanup.py -v` passed: 11 passed, 1 warning.
- `cd server && uv run ruff check wake tests/test_wake_gate.py tests/test_transcript_cleanup.py` passed.
- `cd server && uv run pyright wake tests/test_wake_gate.py tests/test_transcript_cleanup.py` passed.
- `cd server && uv run python - <<'PY' ... OpenWakeWordDetector(Path('models/mave.onnx')) ... PY` passed and downloaded openWakeWord runtime resources: `melspectrogram.onnx`, `embedding_model.onnx`, and `silero_vad.onnx`.

Summary:
- Added safe openWakeWord runtime resource bootstrap before constructing `Model`, downloading only required feature/VAD resources and raising `OpenWakeWordResourceError` on failure.
- Normalized `Model.predict()` output to `dict[str, float]`, including tuple result handling.
- Added `WakePhraseTranscriptCleaner` FrameProcessor to strip wake phrases from downstream `TranscriptionFrame`s while preserving existing `strip_wake_phrase` behavior.
- Added focused tests for resource bootstrap, clear failures, predict normalization, and transcript cleaner behavior.

Concerns:
- First run in a fresh environment still requires network access to fetch openWakeWord feature/VAD resources if absent.

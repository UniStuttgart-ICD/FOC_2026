# Voice Mod Lab Persona + Gemini Preview

Success criteria:
- Gemini Live TTS profiles can generate clean preview audio instead of showing "Unsupported TTS provider".
- Voice Mod Lab loads the speaking agent persona from prompt files, not duplicated UI text.
- The UI explains which prompt text affects agent persona and speech delivery.
- Tests, lint, type checks, and browser verification pass.

Plan:
1. Add tests for Gemini Live preview synthesis, default speech delivery, and persona endpoint.
2. Implement Gemini Live preview adapter and prompt-derived persona API.
3. Render the persona panel in Voice Mod Lab.
4. Verify with pytest, ruff, pyright, and playwright-cli.

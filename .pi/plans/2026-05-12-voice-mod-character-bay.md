# Voice Mod Character Bay

Success criteria:
- Preset bay becomes a Character Bay with sci-fi archetypes, not raw expert settings.
- Character presets include protocol_droid and masked_breather styles without exact character cloning.
- Voice settings add body_shift and breath_mix as compact, high-impact controls.
- DSP preserves audio length and changes audio when body_shift or breath_mix is enabled.
- Tests, lint/type checks, and browser verification pass.

Plan:
1. Add failing tests for preset names, settings serialization, DSP behavior, and UI labels.
2. Add body_shift and breath_mix to settings validation and DSP.
3. Replace utility preset set with character presets.
4. Update UI copy/macros and verify.

# Live Wake And Default Agent Fix

## Reason

The live log shows wake candidates with high model scores were rejected only because `min_wake_rms=35.0` and `min_wake_peak=100` were too high. The same run also shows the default `hybrid_low_latency` profile correctly using `agent=openai_api`, which must stay on the OpenAI API-key backend with `OPENAI_API_KEY`.

## Plan

- Add profile tests proving the bundled default accepts the logged quiet wake candidate envelope.
- Keep `hybrid_low_latency` on `openai_api`.
- Lower bundled openwakeword score/audio guards enough for conversational wake attempts.
- Ignore standalone `base` transcripts as likely wake-word mistranscriptions.
- Run targeted profile, wake, processor, and LangGraph tests.

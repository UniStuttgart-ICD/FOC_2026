from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceRuntimeParts:
    transport_input: object
    voice_command_audio: object | None
    stt: object
    voice_command_transcript: object | None
    user_aggregator: object
    agent_turn: object
    tts: object
    voice_modulation: object | None
    wake_tone: object | None
    transport_output: object
    assistant_aggregator: object
    bot_speech_output: object | None = None


def ordered_voice_runtime_processors(parts: VoiceRuntimeParts) -> list[object]:
    processors = [parts.transport_input]
    if parts.voice_command_audio is not None:
        processors.append(parts.voice_command_audio)
    processors.append(parts.stt)
    if parts.voice_command_transcript is not None:
        processors.append(parts.voice_command_transcript)
    processors.extend(
        [
            parts.user_aggregator,
            parts.agent_turn,
            parts.tts,
        ]
    )
    if parts.voice_modulation is not None:
        processors.append(parts.voice_modulation)
    if parts.bot_speech_output is not None:
        processors.append(parts.bot_speech_output)
    if parts.wake_tone is not None:
        processors.append(parts.wake_tone)
    processors.extend([parts.transport_output, parts.assistant_aggregator])
    return processors

from voice_runtime.assembly import VoiceRuntimeParts, ordered_voice_runtime_processors


def test_orders_voice_runtime_processors_with_wake_adapters():
    parts = VoiceRuntimeParts(
        transport_input="transport.input",
        voice_command_audio="wake.audio",
        stt="stt",
        voice_command_transcript="wake.transcript",
        user_aggregator="user_aggregator",
        agent_turn="agent_turn",
        tts="tts",
        transport_output="transport.output",
        assistant_aggregator="assistant_aggregator",
    )

    assert ordered_voice_runtime_processors(parts) == [
        "transport.input",
        "wake.audio",
        "stt",
        "wake.transcript",
        "user_aggregator",
        "agent_turn",
        "tts",
        "transport.output",
        "assistant_aggregator",
    ]


def test_orders_voice_runtime_processors_without_wake_adapters():
    parts = VoiceRuntimeParts(
        transport_input="transport.input",
        voice_command_audio=None,
        stt="stt",
        voice_command_transcript=None,
        user_aggregator="user_aggregator",
        agent_turn="agent_turn",
        tts="tts",
        transport_output="transport.output",
        assistant_aggregator="assistant_aggregator",
    )

    assert ordered_voice_runtime_processors(parts) == [
        "transport.input",
        "stt",
        "user_aggregator",
        "agent_turn",
        "tts",
        "transport.output",
        "assistant_aggregator",
    ]

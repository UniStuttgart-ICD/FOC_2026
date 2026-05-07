from __future__ import annotations

import numpy as np
import pytest

from voice_modulation.dsp import (
    VoiceModulationDspError,
    pcm16_rms,
    process_pcm16,
)
from voice_modulation.settings import VoiceModulationSettings


def sine_pcm16(sample_rate: int = 24000, hz: float = 440.0, seconds: float = 0.1) -> bytes:
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    samples = np.sin(2.0 * np.pi * hz * t) * 0.25
    return np.asarray(samples * 32767.0, dtype=np.int16).tobytes()


def unique_sample_count(audio: bytes) -> int:
    return len(np.unique(np.frombuffer(audio, dtype=np.int16)))


def test_disabled_settings_return_original_bytes() -> None:
    audio = sine_pcm16()

    processed = process_pcm16(
        audio,
        sample_rate=24000,
        num_channels=1,
        settings=VoiceModulationSettings(enabled=False),
    )

    assert processed == audio


def test_wet_mix_zero_returns_original_bytes() -> None:
    audio = sine_pcm16()

    processed = process_pcm16(
        audio,
        sample_rate=24000,
        num_channels=1,
        settings=VoiceModulationSettings(enabled=True, gain_db=6.0, wet_mix=0.0),
    )

    assert processed == audio


def test_gain_increases_rms() -> None:
    audio = sine_pcm16()

    processed = process_pcm16(
        audio,
        sample_rate=24000,
        num_channels=1,
        settings=VoiceModulationSettings(enabled=True, gain_db=6.0),
    )

    assert pcm16_rms(processed) > pcm16_rms(audio)


def test_bit_depth_reduces_unique_sample_values() -> None:
    audio = sine_pcm16()

    processed = process_pcm16(
        audio,
        sample_rate=24000,
        num_channels=1,
        settings=VoiceModulationSettings(enabled=True, bit_depth=4),
    )

    assert unique_sample_count(processed) < unique_sample_count(audio)


def test_ring_modulation_changes_bytes_and_preserves_length() -> None:
    audio = sine_pcm16()

    processed = process_pcm16(
        audio,
        sample_rate=24000,
        num_channels=1,
        settings=VoiceModulationSettings(enabled=True, ring_mod_hz=40.0),
    )

    assert processed != audio
    assert len(processed) == len(audio)


def test_tremolo_changes_bytes_and_preserves_length() -> None:
    audio = sine_pcm16()

    processed = process_pcm16(
        audio,
        sample_rate=24000,
        num_channels=1,
        settings=VoiceModulationSettings(
            enabled=True,
            tremolo_hz=5.0,
            tremolo_depth=0.5,
        ),
    )

    assert processed != audio
    assert len(processed) == len(audio)


def test_pitch_shift_changes_bytes_and_preserves_length() -> None:
    audio = sine_pcm16(seconds=0.25)

    processed = process_pcm16(
        audio,
        sample_rate=24000,
        num_channels=1,
        settings=VoiceModulationSettings(enabled=True, pitch_shift_semitones=-5.0),
    )

    assert processed != audio
    assert len(processed) == len(audio)


def test_chorus_echo_and_noise_change_bytes_and_preserve_length() -> None:
    audio = sine_pcm16(seconds=0.25)

    processed = process_pcm16(
        audio,
        sample_rate=24000,
        num_channels=1,
        settings=VoiceModulationSettings(
            enabled=True,
            chorus_rate_hz=0.8,
            chorus_depth_ms=16.0,
            chorus_mix=0.4,
            echo_delay_ms=35.0,
            echo_feedback=0.25,
            echo_mix=0.25,
            noise_mix=0.01,
        ),
    )

    assert processed != audio
    assert len(processed) == len(audio)


def test_filters_change_bytes_and_preserve_length() -> None:
    audio = sine_pcm16()

    processed = process_pcm16(
        audio,
        sample_rate=24000,
        num_channels=1,
        settings=VoiceModulationSettings(
            enabled=True,
            low_cut_hz=300.0,
            high_cut_hz=3000.0,
        ),
    )

    assert processed != audio
    assert len(processed) == len(audio)


def test_invalid_channel_or_byte_alignment_raises() -> None:
    audio = sine_pcm16()

    with pytest.raises(VoiceModulationDspError):
        process_pcm16(
            audio,
            sample_rate=24000,
            num_channels=0,
            settings=VoiceModulationSettings(enabled=True),
        )

    with pytest.raises(VoiceModulationDspError):
        process_pcm16(
            audio + b"\0",
            sample_rate=24000,
            num_channels=1,
            settings=VoiceModulationSettings(enabled=True),
        )

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from voice_modulation.settings import VoiceModulationSettings


class VoiceModulationDspError(ValueError):
    """Raised when PCM audio cannot be processed."""


@dataclass
class VoiceModulationState:
    ring_phase: float = 0.0
    tremolo_phase: float = 0.0
    low_cut_last: NDArray[np.float32] | None = None
    high_cut_last: NDArray[np.float32] | None = None
    high_cut_prev_input: NDArray[np.float32] | None = None

    def reset(self) -> None:
        self.ring_phase = 0.0
        self.tremolo_phase = 0.0
        self.low_cut_last = None
        self.high_cut_last = None
        self.high_cut_prev_input = None


def process_pcm16(
    audio: bytes,
    *,
    sample_rate: int,
    num_channels: int,
    settings: VoiceModulationSettings,
    state: VoiceModulationState | None = None,
) -> bytes:
    _validate_pcm16(audio, sample_rate=sample_rate, num_channels=num_channels)
    if not settings.enabled or settings.wet_mix == 0.0:
        return audio

    dsp_state = state or VoiceModulationState()
    samples = _pcm16_to_float_frames(audio, num_channels)
    if samples.size == 0:
        return audio

    dry = samples.copy()
    samples *= np.float32(10.0 ** (settings.gain_db / 20.0))

    if settings.low_cut_hz > 0.0:
        samples = _high_pass(samples, sample_rate, settings.low_cut_hz, dsp_state)
    if settings.high_cut_hz > 0.0:
        samples = _low_pass(samples, sample_rate, settings.high_cut_hz, dsp_state)
    if settings.drive > 0.0:
        drive = np.float32(1.0 + settings.drive * 8.0)
        samples = np.asarray(np.tanh(samples * drive) / np.tanh(drive), dtype=np.float32)
    if settings.bit_depth < 16:
        samples = _bit_crush(samples, settings.bit_depth)
    if settings.ring_mod_hz > 0.0:
        samples = _ring_mod(samples, sample_rate, settings.ring_mod_hz, dsp_state)
    if settings.tremolo_hz > 0.0 and settings.tremolo_depth > 0.0:
        samples = _tremolo(
            samples,
            sample_rate,
            settings.tremolo_hz,
            settings.tremolo_depth,
            dsp_state,
        )

    wet_mix = np.float32(settings.wet_mix)
    samples = (dry * (np.float32(1.0) - wet_mix)) + (samples * wet_mix)
    clip_peak = np.float32(0.98 if settings.limiter else 1.0)
    samples = np.asarray(np.clip(samples, -clip_peak, clip_peak), dtype=np.float32)
    return _float_frames_to_pcm16(samples)


def pcm16_rms(audio: bytes) -> float:
    if len(audio) % 2 != 0:
        raise VoiceModulationDspError("PCM16 audio length must be byte-aligned")
    if not audio:
        return 0.0
    samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / np.float32(32768.0)
    return float(np.sqrt(np.mean(np.square(samples))))


def _validate_pcm16(audio: bytes, *, sample_rate: int, num_channels: int) -> None:
    if sample_rate <= 0:
        raise VoiceModulationDspError("sample_rate must be greater than 0")
    if num_channels < 1:
        raise VoiceModulationDspError("num_channels must be at least 1")
    frame_width = 2 * num_channels
    if len(audio) % frame_width != 0:
        raise VoiceModulationDspError("PCM16 audio length must align to channel frames")


def _pcm16_to_float_frames(audio: bytes, num_channels: int) -> NDArray[np.float32]:
    samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
    return (samples / np.float32(32768.0)).reshape(-1, num_channels)


def _float_frames_to_pcm16(samples: NDArray[np.float32]) -> bytes:
    pcm = np.rint(samples.reshape(-1) * np.float32(32767.0))
    return np.asarray(pcm, dtype=np.int16).tobytes()


def _ensure_channel_state(
    current: NDArray[np.float32] | None,
    num_channels: int,
) -> NDArray[np.float32]:
    if current is not None and current.shape == (num_channels,):
        return current.astype(np.float32, copy=True)
    return np.zeros(num_channels, dtype=np.float32)


def _high_pass(
    samples: NDArray[np.float32],
    sample_rate: int,
    cutoff_hz: float,
    state: VoiceModulationState,
) -> NDArray[np.float32]:
    cutoff = min(cutoff_hz, sample_rate * 0.45)
    rc = 1.0 / (2.0 * np.pi * cutoff)
    dt = 1.0 / sample_rate
    alpha = np.float32(rc / (rc + dt))
    output = np.empty_like(samples)
    prev_output = _ensure_channel_state(state.low_cut_last, samples.shape[1])
    prev_input = _ensure_channel_state(state.high_cut_prev_input, samples.shape[1])

    for index, row in enumerate(samples):
        current = alpha * (prev_output + row - prev_input)
        output[index] = current
        prev_output = current
        prev_input = row

    state.low_cut_last = prev_output.copy()
    state.high_cut_prev_input = prev_input.copy()
    return output


def _low_pass(
    samples: NDArray[np.float32],
    sample_rate: int,
    cutoff_hz: float,
    state: VoiceModulationState,
) -> NDArray[np.float32]:
    cutoff = min(cutoff_hz, sample_rate * 0.45)
    rc = 1.0 / (2.0 * np.pi * cutoff)
    dt = 1.0 / sample_rate
    alpha = np.float32(dt / (rc + dt))
    output = np.empty_like(samples)
    prev_output = _ensure_channel_state(state.high_cut_last, samples.shape[1])

    for index, row in enumerate(samples):
        current = prev_output + (alpha * (row - prev_output))
        output[index] = current
        prev_output = current

    state.high_cut_last = prev_output.copy()
    return output


def _bit_crush(samples: NDArray[np.float32], bit_depth: int) -> NDArray[np.float32]:
    scale = np.float32((2 ** (bit_depth - 1)) - 1)
    return np.asarray(np.rint(samples * scale) / scale, dtype=np.float32)


def _ring_mod(
    samples: NDArray[np.float32],
    sample_rate: int,
    hz: float,
    state: VoiceModulationState,
) -> NDArray[np.float32]:
    phases = _phases(samples.shape[0], sample_rate, hz, state.ring_phase)
    state.ring_phase = _next_phase(phases[-1], sample_rate, hz) if phases.size else state.ring_phase
    modulator = np.sin(phases, dtype=np.float32).reshape(-1, 1)
    return np.asarray(samples * modulator, dtype=np.float32)


def _tremolo(
    samples: NDArray[np.float32],
    sample_rate: int,
    hz: float,
    depth: float,
    state: VoiceModulationState,
) -> NDArray[np.float32]:
    phases = _phases(samples.shape[0], sample_rate, hz, state.tremolo_phase)
    state.tremolo_phase = (
        _next_phase(phases[-1], sample_rate, hz) if phases.size else state.tremolo_phase
    )
    wave = (np.sin(phases, dtype=np.float32) + np.float32(1.0)) * np.float32(0.5)
    modulator = (np.float32(1.0) - np.float32(depth)) + (np.float32(depth) * wave)
    return np.asarray(samples * modulator.reshape(-1, 1), dtype=np.float32)


def _phases(
    length: int,
    sample_rate: int,
    hz: float,
    start_phase: float,
) -> NDArray[np.float32]:
    increment = np.float32((2.0 * np.pi * hz) / sample_rate)
    return np.asarray(
        np.float32(start_phase) + (np.arange(length, dtype=np.float32) * increment),
        dtype=np.float32,
    )


def _next_phase(last_phase: np.float32, sample_rate: int, hz: float) -> float:
    increment = np.float32((2.0 * np.pi * hz) / sample_rate)
    return float((last_phase + increment) % np.float32(2.0 * np.pi))

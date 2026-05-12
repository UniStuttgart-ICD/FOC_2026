from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from voice_modulation.settings import VoiceModulationSettings


class VoiceModulationDspError(ValueError):
    """Raised when PCM audio cannot be processed."""


@dataclass
class VoiceModulationState:
    ring_phase: float = 0.0
    tremolo_phase: float = 0.0
    chorus_phase: float = 0.0
    low_cut_last: NDArray[np.float32] | None = None
    high_cut_last: NDArray[np.float32] | None = None
    high_cut_prev_input: NDArray[np.float32] | None = None
    echo_buffer: NDArray[np.float32] | None = None
    echo_index: int = 0
    chorus_buffer: NDArray[np.float32] | None = None
    chorus_index: int = 0
    body_low_last: NDArray[np.float32] | None = None
    breath_phase: float = 0.0
    breath_last: NDArray[np.float32] | None = None
    noise_rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def reset(self) -> None:
        self.ring_phase = 0.0
        self.tremolo_phase = 0.0
        self.chorus_phase = 0.0
        self.low_cut_last = None
        self.high_cut_last = None
        self.high_cut_prev_input = None
        self.echo_buffer = None
        self.echo_index = 0
        self.chorus_buffer = None
        self.chorus_index = 0
        self.body_low_last = None
        self.breath_phase = 0.0
        self.breath_last = None
        self.noise_rng = np.random.default_rng()


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
    if settings.pitch_shift_semitones != 0.0:
        samples = _granular_pitch_shift(samples, sample_rate, settings.pitch_shift_semitones)
    if settings.body_shift != 0.0:
        samples = _body_shift(samples, sample_rate, settings.body_shift, dsp_state)
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
    if settings.chorus_mix > 0.0 and settings.chorus_depth_ms > 0.0:
        samples = _chorus(
            samples,
            sample_rate,
            settings.chorus_rate_hz,
            settings.chorus_depth_ms,
            settings.chorus_mix,
            dsp_state,
        )
    if settings.echo_mix > 0.0 and settings.echo_delay_ms > 0.0:
        samples = _echo(
            samples,
            sample_rate,
            settings.echo_delay_ms,
            settings.echo_feedback,
            settings.echo_mix,
            dsp_state,
        )
    if settings.noise_mix > 0.0:
        samples = _noise(samples, settings.noise_mix, dsp_state)
    if settings.breath_mix > 0.0:
        samples = _breath_noise(samples, sample_rate, settings.breath_mix, dsp_state)

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


def _body_shift(
    samples: NDArray[np.float32],
    sample_rate: int,
    amount: float,
    state: VoiceModulationState,
) -> NDArray[np.float32]:
    depth = np.float32(abs(amount))
    cutoff = 780.0 if amount < 0.0 else 1350.0
    low, state.body_low_last = _low_pass_rows(
        samples,
        sample_rate,
        cutoff,
        state.body_low_last,
    )
    if amount < 0.0:
        output = (samples * (np.float32(1.0) - (depth * np.float32(0.25)))) + (
            low * depth * np.float32(0.75)
        )
    else:
        high = samples - low
        output = (samples * (np.float32(1.0) - (depth * np.float32(0.35)))) + (
            high * depth * np.float32(0.9)
        )
    return np.asarray(np.clip(output, -1.0, 1.0), dtype=np.float32)


def _low_pass_rows(
    samples: NDArray[np.float32],
    sample_rate: int,
    cutoff_hz: float,
    previous: NDArray[np.float32] | None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    cutoff = min(cutoff_hz, sample_rate * 0.45)
    rc = 1.0 / (2.0 * np.pi * cutoff)
    dt = 1.0 / sample_rate
    alpha = np.float32(dt / (rc + dt))
    output = np.empty_like(samples)
    prev_output = _ensure_channel_state(previous, samples.shape[1])

    for index, row in enumerate(samples):
        current = prev_output + (alpha * (row - prev_output))
        output[index] = current
        prev_output = current

    return output, prev_output.copy()


def _bit_crush(samples: NDArray[np.float32], bit_depth: int) -> NDArray[np.float32]:
    scale = np.float32((2 ** (bit_depth - 1)) - 1)
    return np.asarray(np.rint(samples * scale) / scale, dtype=np.float32)


def _granular_pitch_shift(
    samples: NDArray[np.float32],
    sample_rate: int,
    semitones: float,
) -> NDArray[np.float32]:
    if samples.shape[0] < 4:
        return samples
    ratio = np.float32(2.0 ** (semitones / 12.0))
    grain_length = min(samples.shape[0], max(64, int(sample_rate * 0.045)))
    hop = max(1, grain_length // 2)
    window = np.hanning(grain_length).astype(np.float32)
    source_index = np.arange(samples.shape[0], dtype=np.float32)
    output = np.zeros_like(samples)
    weights = np.zeros((samples.shape[0], 1), dtype=np.float32)

    for start in range(0, samples.shape[0], hop):
        end = min(samples.shape[0], start + grain_length)
        size = end - start
        positions = np.asarray(
            start + (np.arange(size, dtype=np.float32) * ratio),
            dtype=np.float32,
        )
        positions = np.clip(positions, 0.0, np.float32(samples.shape[0] - 1))
        shifted = np.empty((size, samples.shape[1]), dtype=np.float32)
        for channel in range(samples.shape[1]):
            shifted[:, channel] = np.interp(
                positions,
                source_index,
                samples[:, channel],
            ).astype(np.float32)
        gain = window[:size].reshape(-1, 1)
        output[start:end] += shifted * gain
        weights[start:end] += gain

    return np.asarray(output / np.maximum(weights, np.float32(1e-6)), dtype=np.float32)


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


def _echo(
    samples: NDArray[np.float32],
    sample_rate: int,
    delay_ms: float,
    feedback: float,
    mix: float,
    state: VoiceModulationState,
) -> NDArray[np.float32]:
    delay_frames = max(1, int(sample_rate * (delay_ms / 1000.0)))
    if state.echo_buffer is None or state.echo_buffer.shape != (delay_frames, samples.shape[1]):
        state.echo_buffer = np.zeros((delay_frames, samples.shape[1]), dtype=np.float32)
        state.echo_index = 0

    output = np.empty_like(samples)
    wet_mix = np.float32(mix)
    dry_mix = np.float32(1.0) - wet_mix
    feedback_gain = np.float32(feedback)

    for index, row in enumerate(samples):
        delayed = state.echo_buffer[state.echo_index].copy()
        output[index] = (row * dry_mix) + (delayed * wet_mix)
        state.echo_buffer[state.echo_index] = row + (delayed * feedback_gain)
        state.echo_index = (state.echo_index + 1) % delay_frames

    return output


def _chorus(
    samples: NDArray[np.float32],
    sample_rate: int,
    rate_hz: float,
    depth_ms: float,
    mix: float,
    state: VoiceModulationState,
) -> NDArray[np.float32]:
    base_delay_frames = max(1, int(sample_rate * 0.012))
    depth_frames = max(1, int(sample_rate * (depth_ms / 1000.0)))
    buffer_length = base_delay_frames + depth_frames + 3
    if state.chorus_buffer is None or state.chorus_buffer.shape != (
        buffer_length,
        samples.shape[1],
    ):
        state.chorus_buffer = np.zeros((buffer_length, samples.shape[1]), dtype=np.float32)
        state.chorus_index = 0

    output = np.empty_like(samples)
    wet_mix = np.float32(mix)
    dry_mix = np.float32(1.0) - wet_mix
    increment = np.float32((2.0 * np.pi * rate_hz) / sample_rate)

    for index, row in enumerate(samples):
        state.chorus_buffer[state.chorus_index] = row
        lfo = (np.sin(np.float32(state.chorus_phase)) + np.float32(1.0)) * np.float32(0.5)
        delay = np.float32(base_delay_frames) + (np.float32(depth_frames) * lfo)
        read_position = (np.float32(state.chorus_index) - delay) % np.float32(buffer_length)
        left_index = int(np.floor(read_position))
        right_index = (left_index + 1) % buffer_length
        fraction = read_position - np.float32(left_index)
        delayed = (
            state.chorus_buffer[left_index] * (np.float32(1.0) - fraction)
        ) + (state.chorus_buffer[right_index] * fraction)
        output[index] = (row * dry_mix) + (delayed * wet_mix)
        state.chorus_phase = float(
            (np.float32(state.chorus_phase) + increment) % np.float32(2.0 * np.pi)
        )
        state.chorus_index = (state.chorus_index + 1) % buffer_length

    return output


def _noise(
    samples: NDArray[np.float32],
    mix: float,
    state: VoiceModulationState,
) -> NDArray[np.float32]:
    noise = state.noise_rng.uniform(-1.0, 1.0, size=samples.shape).astype(np.float32)
    return np.asarray(samples + (noise * np.float32(mix)), dtype=np.float32)


def _breath_noise(
    samples: NDArray[np.float32],
    sample_rate: int,
    mix: float,
    state: VoiceModulationState,
) -> NDArray[np.float32]:
    noise = state.noise_rng.uniform(-1.0, 1.0, size=samples.shape).astype(np.float32)
    filtered, state.breath_last = _low_pass_rows(
        noise,
        sample_rate,
        950.0,
        state.breath_last,
    )
    phases = _phases(samples.shape[0], sample_rate, 0.55, state.breath_phase)
    state.breath_phase = (
        _next_phase(phases[-1], sample_rate, 0.55) if phases.size else state.breath_phase
    )
    envelope = (
        np.float32(0.25)
        + (np.float32(0.75) * ((np.sin(phases, dtype=np.float32) + np.float32(1.0)) * 0.5))
    ).reshape(-1, 1)
    return np.asarray(samples + (filtered * envelope * np.float32(mix)), dtype=np.float32)


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

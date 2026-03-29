from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass(slots=True)
class AudioFeatures:
    source_path: Path
    sample_rate: int
    duration_seconds: float
    audio: np.ndarray
    chroma_12: np.ndarray
    pitch_class_12: np.ndarray
    pitch_class_24: np.ndarray
    combined_12: np.ndarray
    voiced_ratio: float
    voiced_frame_count: int
    tuning_offset: float


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    total = float(np.sum(vector))
    if total <= 0:
        return np.zeros_like(vector, dtype=float)
    return vector / total


def load_audio_features(
    path: str | Path,
    *,
    sample_rate: int = 22_050,
    hop_length: int = 512,
) -> AudioFeatures:
    source_path = Path(path)
    audio, sr = librosa.load(source_path, sr=sample_rate, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=35)
    if audio.size == 0:
        raise ValueError(f"No usable audio was found in {source_path}.")

    harmonic = librosa.effects.harmonic(audio)
    tuning_offset = float(librosa.estimate_tuning(y=harmonic, sr=sr))

    chroma = librosa.feature.chroma_cqt(
        y=harmonic,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=36,
    )
    chroma_12 = normalize(chroma.mean(axis=1))

    f0, voiced_flag, voiced_prob = librosa.pyin(
        harmonic,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=hop_length,
    )

    pitch_class_12 = histogram_from_f0(f0, voiced_prob, 12, tuning_offset=tuning_offset)
    pitch_class_24 = histogram_from_f0(f0, voiced_prob, 24, tuning_offset=tuning_offset)
    voiced_ratio = float(np.nanmean(voiced_flag.astype(float))) if voiced_flag is not None else 0.0
    voiced_frame_count = int(np.count_nonzero(np.nan_to_num(voiced_prob, nan=0.0) > 0.1))

    if pitch_class_12.sum() > 0:
        combined_12 = normalize(0.7 * chroma_12 + 0.3 * pitch_class_12)
    else:
        combined_12 = chroma_12

    return AudioFeatures(
        source_path=source_path,
        sample_rate=sr,
        duration_seconds=float(audio.shape[0] / sr),
        audio=audio,
        chroma_12=chroma_12,
        pitch_class_12=pitch_class_12,
        pitch_class_24=pitch_class_24,
        combined_12=combined_12,
        voiced_ratio=voiced_ratio,
        voiced_frame_count=voiced_frame_count,
        tuning_offset=tuning_offset,
    )


def histogram_from_f0(
    f0: np.ndarray | None,
    voiced_prob: np.ndarray | None,
    bins: int,
    *,
    tuning_offset: float = 0.0,
) -> np.ndarray:
    if f0 is None:
        return np.zeros(bins, dtype=float)

    f0 = np.asarray(f0, dtype=float)
    valid = np.isfinite(f0) & (f0 > 0)
    if not np.any(valid):
        return np.zeros(bins, dtype=float)

    midi = librosa.hz_to_midi(f0[valid]) - tuning_offset
    weights = np.ones_like(midi, dtype=float)
    if voiced_prob is not None:
        voiced_prob = np.asarray(voiced_prob, dtype=float)
        weights = np.nan_to_num(voiced_prob[valid], nan=0.0)

    if bins == 12:
        indices = np.mod(np.round(midi).astype(int), 12)
    elif bins == 24:
        indices = np.mod(np.round(midi * 2.0).astype(int), 24)
    else:
        raise ValueError(f"Unsupported bin count: {bins}")

    histogram = np.bincount(indices, weights=weights, minlength=bins).astype(float)
    return normalize(histogram)


def lift_12_to_24(pitch_class_12: np.ndarray) -> np.ndarray:
    lifted = np.zeros(24, dtype=float)
    lifted[::2] = normalize(pitch_class_12)
    return normalize(lifted)


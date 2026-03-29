from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .catalog import ScaleTemplate, build_default_templates, note_name_for_bin
from .examples import lookup_example
from .features import AudioFeatures, lift_12_to_24, load_audio_features, normalize


@dataclass(slots=True)
class ScaleMatch:
    template: ScaleTemplate
    tonic_bin: int
    tonic_label: str
    score: float
    cosine_similarity: float
    support_f1: float
    feature_space: str


@dataclass(slots=True)
class AnalysisResult:
    source_path: Path | None
    features: AudioFeatures | None
    matches: list[ScaleMatch]
    culture_scores: dict[str, float]

    @property
    def best_match(self) -> ScaleMatch:
        return self.matches[0]

    @property
    def confidence_note(self) -> str:
        score = self.best_match.score
        if score >= 0.88:
            return "Strong within-catalog match. This is still the closest supported family, not a definitive ethnomusicological identification."
        if score >= 0.72:
            return "Moderate within-catalog match. Shared pitch material across traditions is possible."
        return "Low-confidence match. The source may fall outside the current catalog or may require phrase-level modeling beyond scale degrees."

    def as_dataframe(self, limit: int = 5) -> pd.DataFrame:
        rows = []
        for match in self.matches[:limit]:
            example = lookup_example(match.template)
            rows.append(
                {
                    "culture": match.template.culture,
                    "family": match.template.family,
                    "scale": match.template.name,
                    "tonic": match.tonic_label,
                    "score": round(match.score, 3),
                    "cosine": round(match.cosine_similarity, 3),
                    "support_f1": round(match.support_f1, 3),
                    "feature_space": match.feature_space,
                    "example_title": example.title if example else "",
                    "example_artist": example.artist if example else "",
                    "youtube_url": example.youtube_url if example else "",
                    "spotify_url": example.spotify_url if example else "",
                    "notes": match.template.description,
                }
            )
        return pd.DataFrame(rows)

    def summary_markdown(self) -> str:
        best = self.best_match
        cultures = ", ".join(f"{culture}: {score:.2f}" for culture, score in self.culture_scores.items())
        source = str(self.source_path) if self.source_path else "in-memory histograms"
        return (
            f"### Best Match\n"
            f"- Source: `{source}`\n"
            f"- Culture: **{best.template.culture}**\n"
            f"- Scale: **{best.template.name}**\n"
            f"- Family: `{best.template.family}`\n"
            f"- Tonic: `{best.tonic_label}`\n"
            f"- Score: `{best.score:.3f}` using `{best.feature_space}`\n"
            f"- Culture scores: {cultures}\n"
            f"- Interpretation: {self.confidence_note}"
        )


class ScaleAnalyzer:
    def __init__(self, templates: Iterable[ScaleTemplate] | None = None) -> None:
        self.templates = list(templates or build_default_templates())

    def supported_cultures(self) -> list[str]:
        return sorted({template.culture for template in self.templates})

    def analyze_file(self, path: str | Path, *, top_k: int = 8) -> AnalysisResult:
        features = load_audio_features(path)
        result = self.classify_histograms(
            pitch_class_12=features.combined_12,
            pitch_class_24=features.pitch_class_24,
            voiced_ratio=features.voiced_ratio,
            voiced_frame_count=features.voiced_frame_count,
            top_k=top_k,
        )
        result.source_path = Path(path)
        result.features = features
        return result

    def classify_histograms(
        self,
        *,
        pitch_class_12: np.ndarray,
        pitch_class_24: np.ndarray | None = None,
        voiced_ratio: float = 0.0,
        voiced_frame_count: int = 0,
        top_k: int = 8,
    ) -> AnalysisResult:
        observed_12 = normalize(np.asarray(pitch_class_12, dtype=float))
        if pitch_class_24 is None or not np.any(pitch_class_24):
            observed_24 = lift_12_to_24(observed_12)
            quarter_tone_confidence = 0.45
        else:
            observed_24 = normalize(np.asarray(pitch_class_24, dtype=float))
            quarter_tone_confidence = min(1.0, voiced_frame_count / 120.0)
        microtonal_energy = float(observed_24[1::2].sum()) if observed_24.size == 24 else 0.0

        matches: list[ScaleMatch] = []
        for template in self.templates:
            observed = observed_24 if template.bins == 24 else observed_12
            best_match = self._best_match_for_template(
                template,
                observed,
                quarter_tone_confidence=quarter_tone_confidence,
                voiced_ratio=voiced_ratio,
                microtonal_energy=microtonal_energy,
            )
            matches.append(best_match)

        matches.sort(key=lambda item: item.score, reverse=True)

        culture_scores_raw: dict[str, float] = {}
        for match in matches:
            current = culture_scores_raw.get(match.template.culture, 0.0)
            culture_scores_raw[match.template.culture] = max(current, match.score)

        total = sum(culture_scores_raw.values()) or 1.0
        culture_scores = {
            culture: score / total
            for culture, score in sorted(culture_scores_raw.items(), key=lambda item: item[1], reverse=True)
        }

        return AnalysisResult(
            source_path=None,
            features=None,
            matches=matches[:top_k],
            culture_scores=culture_scores,
        )

    def _best_match_for_template(
        self,
        template: ScaleTemplate,
        observed: np.ndarray,
        *,
        quarter_tone_confidence: float,
        voiced_ratio: float,
        microtonal_energy: float,
    ) -> ScaleMatch:
        best: ScaleMatch | None = None
        tonic_count = template.bins
        microtonal_signature = float(np.clip((microtonal_energy - 0.12) / 0.28, 0.0, 1.0))

        for tonic in range(tonic_count):
            active_bins = np.array([(tonic + interval) % template.bins for interval in template.intervals], dtype=int)
            template_vector = build_template_vector(active_bins, template.bins)
            cosine_similarity = cosine_score(observed, template_vector)
            support_f1 = support_score(observed, active_bins)
            score = 0.6 * cosine_similarity + 0.4 * support_f1

            if template.bins == 24:
                score *= 0.75 + 0.25 * quarter_tone_confidence
                score *= 0.8 + 0.2 * min(1.0, voiced_ratio / 0.25)
                score = score + (1.0 - score) * 0.4 * microtonal_signature
            else:
                score *= 1.0 - 0.35 * microtonal_signature * quarter_tone_confidence

            candidate = ScaleMatch(
                template=template,
                tonic_bin=tonic,
                tonic_label=note_name_for_bin(tonic, template.bins),
                score=float(np.clip(score, 0.0, 1.0)),
                cosine_similarity=float(cosine_similarity),
                support_f1=float(support_f1),
                feature_space="quarter-tone histogram" if template.bins == 24 else "12-tone profile",
            )
            if best is None or candidate.score > best.score:
                best = candidate

        assert best is not None
        return best


def build_template_vector(active_bins: np.ndarray, bins: int) -> np.ndarray:
    vector = np.full(bins, 0.02, dtype=float)
    neighbor_weight = 0.18 if bins == 24 else 0.08
    for bin_index in active_bins:
        vector[bin_index % bins] = 1.0
        vector[(bin_index - 1) % bins] = max(vector[(bin_index - 1) % bins], neighbor_weight)
        vector[(bin_index + 1) % bins] = max(vector[(bin_index + 1) % bins], neighbor_weight)
    return normalize(vector)


def cosine_score(observed: np.ndarray, template_vector: np.ndarray) -> float:
    denominator = float(np.linalg.norm(observed) * np.linalg.norm(template_vector))
    if denominator == 0:
        return 0.0
    return float(np.dot(observed, template_vector) / denominator)


def support_score(observed: np.ndarray, active_bins: np.ndarray) -> float:
    threshold = max(0.06, float(observed.mean()) * (1.2 if observed.shape[0] == 24 else 1.0))
    observed_support = set(np.flatnonzero(observed >= threshold))
    template_support = set(int(item) for item in active_bins.tolist())

    if not observed_support:
        return 0.0

    intersection = len(observed_support & template_support)
    if intersection == 0:
        return 0.0

    precision = intersection / len(observed_support)
    recall = intersection / len(template_support)
    return float((2.0 * precision * recall) / (precision + recall))

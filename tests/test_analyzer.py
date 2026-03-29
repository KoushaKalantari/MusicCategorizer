from __future__ import annotations

import numpy as np

from music_categorizer import ScaleAnalyzer
from music_categorizer.catalog import build_default_templates
from music_categorizer.examples import lookup_example
from music_categorizer.notebook_ui import choose_recording_sample_rate


def histogram(intervals: tuple[int, ...], bins: int) -> np.ndarray:
    values = np.zeros(bins, dtype=float)
    values[list(intervals)] = 1.0
    return values / values.sum()


def test_indian_bilawal_template_wins_on_major_profile() -> None:
    analyzer = ScaleAnalyzer()
    result = analyzer.classify_histograms(
        pitch_class_12=histogram((0, 2, 4, 5, 7, 9, 11), 12),
        pitch_class_24=None,
        voiced_ratio=0.8,
        voiced_frame_count=120,
    )

    assert result.best_match.template.culture == "Indian"
    assert result.best_match.template.name == "Bilawal"
    assert result.best_match.tonic_label == "C"


def test_persian_segah_template_wins_on_quarter_tone_profile() -> None:
    analyzer = ScaleAnalyzer()
    result = analyzer.classify_histograms(
        pitch_class_12=histogram((0, 2, 4, 5, 7, 9, 10), 12),
        pitch_class_24=histogram((0, 4, 7, 10, 14, 17, 20), 24),
        voiced_ratio=0.9,
        voiced_frame_count=240,
    )

    assert result.best_match.template.culture == "Persian"
    assert result.best_match.template.name == "Segah"
    assert result.best_match.tonic_label == "C"


def test_examples_are_attached_to_dataframe_rows() -> None:
    analyzer = ScaleAnalyzer()
    result = analyzer.classify_histograms(
        pitch_class_12=histogram((0, 2, 4, 5, 7, 9, 11), 12),
        pitch_class_24=None,
        voiced_ratio=0.8,
        voiced_frame_count=120,
    )

    example = lookup_example(result.best_match.template)
    frame = result.as_dataframe(limit=1)

    assert example is not None
    assert frame.loc[0, "example_title"] == example.title
    assert "youtube.com" in frame.loc[0, "youtube_url"]


def test_choose_recording_sample_rate_uses_safe_fallbacks() -> None:
    assert choose_recording_sample_rate(None) == 44_100
    assert choose_recording_sample_rate(48_000.0) == 48_000
    assert choose_recording_sample_rate(192_000.0) == 48_000


def test_default_catalog_spans_multiple_cultures() -> None:
    templates = build_default_templates()
    cultures = {template.culture for template in templates}

    assert {"Persian", "Indian", "Chinese", "Arabic", "Turkish", "Japanese", "Western"} <= cultures

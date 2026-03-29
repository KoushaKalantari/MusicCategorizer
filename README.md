# Music Categorizer

This repository contains a notebook-first prototype for estimating which scale family a recording most closely matches and which culture that scale library comes from. The current template library focuses on:

- Persian dastgah families
- Indian thaats
- Chinese pentatonic modes
- Arabic maqamat
- Turkish makam approximations
- Japanese pentatonic modes
- Western modal and scalar families

The workflow is designed for melody-forward audio. It works best on solo voice or lead-instrument recordings, or on mixes where the main melodic line is clearly dominant.

## Short Paper-Style Overview

### Title

Heuristic Estimation of Cultural Scale Families from Monophonic or Melody-Dominant Audio

### Abstract

This project implements a lightweight computational pipeline for estimating the most likely cultural scale family present in an audio recording. Rather than training a large statistical model, the system uses signal-processing features and compares them against a curated library of scale templates drawn from Persian dastgah families, Indian thaats, and Chinese pentatonic modes. The method is intended as an interpretable prototype for exploratory listening, educational analysis, and rapid musical sketching. It is most reliable when applied to melody-forward recordings in which the tonic and scale degrees are clearly represented over time.

### Objective

The main objective is to answer a practical question:

"Given an audio clip, which supported cultural scale family best explains the observed pitch content?"

The current system does not attempt full maqam, raga, or dastgah performance recognition in the musicological sense. Instead, it estimates the closest matching pitch-material family under a transparent template-matching framework.

### Method Summary

The method can be summarized in four stages:

1. Audio is loaded and trimmed to remove leading or trailing silence.
2. Harmonic and pitch-based descriptors are extracted from the signal.
3. Observed pitch distributions are compared with cultural scale templates.
4. The system returns the highest-scoring scale, tonic estimate, and culture-level score breakdown.

### Signal Representation

Two complementary pitch representations are used:

- A 12-tone chroma profile derived from constant-Q analysis, used to summarize semitone-level pitch emphasis.
- A pitch-class histogram derived from `pyin` fundamental-frequency tracking, used to capture the distribution of voiced notes over time.

For Persian templates, the system also constructs a 24-bin pitch-class representation so that quarter-tone behavior can be approximated. This is not a full microtonal transcription system, but it gives the matcher a way to distinguish some neutral or quarter-tone pitch centers from nearby 12-tone alternatives.

### Template Library

The current scale library is intentionally interpretable and now broader than the first prototype. It includes:

- Persian dastgah-family approximations represented in a 24-bin octave
- Indian thaats represented in a 12-bin octave
- Chinese pentatonic mode rotations represented in a 12-bin octave
- Arabic maqam approximations represented in a 24-bin octave
- Turkish makam approximations represented in a 24-bin octave
- Japanese pentatonic families represented in a 12-bin octave
- Western modal and scalar families represented in a 12-bin octave

Each template is encoded as a set of scale degrees relative to a tonic. During inference, the system tests all tonic rotations and keeps the best-scoring alignment.

### Inference Procedure

For each candidate template, the observed pitch profile is compared against an idealized template vector. The scoring procedure combines:

- cosine similarity, which measures overall shape agreement between observed and expected pitch emphasis
- a support-style score, which measures how well the set of active observed pitch classes overlaps with the template degrees

When quarter-tone evidence is present, Persian 24-bin templates receive additional weight, while nearby 12-tone competitors are reduced accordingly. This helps prevent clearly microtonal material from collapsing into a purely semitone-based interpretation.

### Interpretation

The output should be interpreted as a ranked hypothesis list rather than a definitive cultural label. In practice:

- a high score suggests strong agreement between the audio's pitch distribution and a template
- close scores across cultures suggest ambiguity or shared pitch material
- the top tonic is the tonic that best aligns the observed distribution to the template under the current heuristic

### Limitations

This method has several important limitations:

- It focuses on pitch content and does not model ornamentation, phrase grammar, cadential behavior, or melodic motion in depth.
- It is stronger at thaat-like or scale-family matching than full tradition-specific identity recognition.
- Dense polyphonic recordings, modulation, drone-heavy textures, and expressive intonation can reduce reliability.
- Persian quarter-tone handling is approximate and should be understood as a coarse computational proxy.

### Practical Conclusion

This repository is best viewed as an interpretable first-pass analysis tool. It is useful for experimentation, prototype building, and educational exploration, especially when the user wants transparent intermediate representations rather than a black-box classifier. The template catalog is now data-driven, so additional traditions can be added by extending `music_categorizer/data/scale_templates.json`.

## What is included

- A reusable Python package in `music_categorizer/`
- An interactive notebook in `culture_scale_lab.ipynb`
- Optional microphone capture from inside the notebook
- A small test layer for the template matcher

## How it works

1. The analyzer loads audio with `librosa`.
2. It extracts a 12-tone chroma profile and a quarter-tone pitch-class histogram.
3. It scores the recording against a curated library of cultural scale templates.
4. It reports the top candidate scales, the most likely tonic, and confidence-style scores.

This is a heuristic classifier, not a trained ethnomusicology model. It can confuse traditions that share pitch material, and it cannot fully resolve raga or dastgah identity from scale degrees alone because phrasing, ornamentation, and melodic motion also matter.

## Setup

Use the virtual environment already in this folder, then install the notebook/runtime dependencies:

```bash
.venv/bin/pip install -r requirements-dev.txt
```

If you want to run the notebook:

```bash
.venv/bin/python run_notebook.py
```

That launcher keeps Jupyter's config and cache inside this project, and it creates a project-local kernel spec that points at `.venv` so the notebook uses the correct interpreter.

If you also want the CLI commands:

```bash
.venv/bin/pip install -e .
```

## Notebook usage

- Upload an audio file with `Choose file`, then click `Analyze uploaded file`.
- Or press `Start recording`, perform into the microphone, and press `Stop recording` when finished.
- The notebook will show:
  - best overall match
  - culture score breakdown
  - top scale candidates
  - representative listening examples for each suggested row
  - audio playback
  - pitch-class plots

The recorder now chooses the microphone device's preferred sample rate automatically when possible, with a safe high-quality fallback. Recording is manually stopped by the user, but it will also auto-stop after 30 seconds to avoid runaway memory use.
Live microphone capture files are treated as temporary artifacts and are deleted automatically after analysis completes.

For the cleanest results, prefer uncompressed or lightly compressed files such as `.wav`, `.flac`, or `.m4a`.

## CLI usage

```bash
.venv/bin/music-scale-analyze path/to/audio.wav
```

## Notes and caveats

- Persian mode estimates use quarter-tone templates adapted to a 24-bin octave, so they are still approximations.
- Indian support is thaat-level, not full raga recognition.
- Chinese support currently targets the five core pentatonic modes.
- Arabic and Turkish support currently use simplified 24-bin approximations rather than full theoretical tuning systems.
- Japanese support currently focuses on a small set of commonly taught pentatonic families.
- Microphone recording uses `sounddevice`, so macOS may prompt for microphone permission the first time you record.
- If you prefer launching Jupyter manually, set repo-local config dirs first or use `music-scale-notebook`.
- Listening examples are curated pointers for familiar recordings or songs associated with the predicted family or scale area; they are illustrative, not authoritative annotations.
- Live recording is optimized for short analysis captures rather than long-form session recording.
- True all-world culture recognition is still an open-ended research goal. Many traditions require phrase structure, ornamentation, tuning nuance, timbre, and corpus-based modeling beyond scale templates alone.

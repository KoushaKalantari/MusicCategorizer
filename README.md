# Music Categorizer

## TL;DR

This project analyzes an uploaded audio file or a short live microphone recording and estimates:

- the closest supported musical culture
- the closest supported scale family or mode
- the most likely tonic
- a ranked list of alternative matches

It is a transparent, heuristic system built from signal-processing features plus a curated template catalog. It currently works from a starter library of templates spanning:

- Persian
- Indian
- Chinese
- Arabic
- Turkish
- Japanese
- Western modal and scalar families

The output should be read as "closest supported match in the current catalog," not as a definitive global musicology classifier.

## Methodology

### Problem framing

The tool is designed to answer a practical question:

"Given a melody-forward recording, which supported scale family best matches its observed pitch content?"

This is intentionally narrower than full cultural or genre recognition. Many traditions depend not only on pitch collections, but also on melodic grammar, ornamentation, register behavior, cadence, rhythm, timbre, and performance practice. This code focuses primarily on pitch structure.

### Core logic

The analysis pipeline is implemented in a small set of modules:

- `music_categorizer/features.py`
  Loads audio, trims silence, extracts harmonic content, computes chroma, and estimates voiced pitch with `pyin`.
- `music_categorizer/catalog.py`
  Loads the template library from `music_categorizer/data/scale_templates.json`.
- `music_categorizer/analyzer.py`
  Rotates each template over possible tonics, scores every alignment, and ranks the results.
- `music_categorizer/notebook_ui.py`
  Provides the notebook interface for file upload, live recording, result display, plots, and listening examples.
- `music_categorizer/notebook_launcher.py`
  Starts Jupyter with project-local config and kernel settings so the notebook uses the correct environment.

### Feature extraction

For each recording, the code builds two complementary pitch representations:

1. A 12-tone chroma profile.
   This captures semitone-level pitch emphasis over time.

2. A pitch-class histogram from estimated fundamental frequency.
   This captures which pitch classes are actually sung or played in voiced frames.

These are combined into a working 12-tone observation profile. For traditions that require microtonal approximation, the system also constructs a 24-bin octave representation.

### Template matching

Each supported scale family is stored as a template with:

- culture
- family
- scale or mode name
- octave resolution (`12` or `24` bins)
- interval pattern relative to tonic

During inference, the analyzer tests each template against all tonic rotations and computes a score using:

- cosine similarity between the observed pitch profile and the template profile
- a support-overlap score that checks whether active pitch bins align with the expected scale degrees

For 24-bin templates, the scorer also considers whether the input appears genuinely microtonal. This helps reduce cases where quarter-tone material is flattened into a nearby 12-tone interpretation.

### Result interpretation

The analyzer returns:

- a best overall match
- normalized culture-level scores
- a ranked set of top candidates
- a tonic estimate

High scores mean the observed pitch distribution resembles a supported template strongly. Lower scores usually mean one of three things:

- the source is ambiguous
- several traditions share similar pitch material
- the music falls outside the current template library

### Scope and limitations

This project is broader than the initial prototype, but it is still not a true "all musical cultures of the world" recognizer. Scale-template matching alone is not enough for that. A more complete system would need:

- phrase-level modeling
- ornament and intonation modeling
- larger and better validated corpora
- culture-specific tuning systems
- learned models beyond a fixed template catalog

So the right claim for this repository is:

"A transparent, extensible scale-family matcher over a growing cross-cultural template catalog."

## Examples

### What the notebook does

The notebook lets you either:

- upload an audio file
- record a short live sample from the microphone

For each run, it shows:

- the top predicted culture and scale family
- culture score breakdown
- top ranked candidate rows
- representative listening examples for each row
- audio playback
- analysis plots

### Example kinds of outcomes

- A melody close to a major-scale collection may land near `Indian / Bilawal` or `Western / Ionian`, depending on how the observed pitch energy aligns with the catalog.
- A melody centered on a Chinese-style anhemitonic pentatonic collection may land near one of the supported Chinese pentatonic modes.
- A recording with strong quarter-tone behavior may push Persian, Arabic, or Turkish 24-bin templates above nearby 12-tone competitors.

### Listening examples

Each suggested row includes a curated listening pointer, usually via YouTube and Spotify search links. These examples are meant to help the user compare the predicted result against familiar repertoire or representative performances. They are illustrative references, not ground-truth labels for the analyzed recording.

## Setup

Install the notebook and runtime dependencies:

```bash
.venv/bin/pip install -r requirements-dev.txt
```

If you also want the CLI entry points:

```bash
.venv/bin/pip install -e .
```

## Running the notebook

Start the notebook with:

```bash
.venv/bin/python launch_scale_lab.py
```

This launcher keeps Jupyter config and cache inside the project and creates a project-local kernel that points at `.venv`.

Open:

```text
music_scale_lab.ipynb
```

## Notebook usage

### Upload flow

1. Click `Choose file`.
2. Select an audio file.
3. Click `Analyze uploaded file`.

### Live recording flow

1. Click `Start recording`.
2. Perform or sing into the microphone.
3. Click `Stop recording`.

The recorder:

- uses the microphone device's preferred sample rate when possible
- falls back to a safe default when needed
- auto-stops after 30 seconds to avoid runaway memory use
- deletes temporary live-recording files after analysis completes

Uploaded files are analyzed through temporary working copies, which are also deleted after analysis. The original uploaded file is left untouched.

For best results, use melody-forward audio such as solo voice, flute, violin, oud, sitar, ney, or similarly exposed lead material.

## CLI usage

```bash
.venv/bin/music-scale-analyze path/to/audio.wav
```

## Extending the catalog

The supported templates are stored in:

```text
music_categorizer/data/scale_templates.json
```

To add more traditions or families, extend that file with additional templates. In practice, the main challenge is not adding rows, but choosing interval patterns that are musically defensible and explicit about approximation.

## Notes

- Persian, Arabic, and Turkish support currently use simplified 24-bin approximations rather than full theoretical tuning systems.
- Indian support is currently thaat-level, not full raga recognition.
- Chinese support currently emphasizes core pentatonic rotations.
- Japanese support currently covers a small pedagogical pentatonic set.
- Western support is included mainly as a useful reference family and overlap baseline.
- Microphone recording uses `sounddevice`, so macOS may ask for microphone permission on first use.
- Low-confidence results should be treated as "unsupported or ambiguous" rather than forced cultural conclusions.

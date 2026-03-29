from __future__ import annotations

import html
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    matplotlib_cache = Path.cwd() / ".cache" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from IPython.display import Audio, HTML as DisplayHTML, Markdown, display
from ipywidgets import (
    Button,
    FileUpload,
    HTML as WidgetHTML,
    HBox,
    Output,
    VBox,
)

from .analyzer import AnalysisResult, ScaleAnalyzer, build_template_vector
from .catalog import NOTE_NAMES_12, note_name_for_bin
from .examples import lookup_example

MAX_RECORDING_SECONDS = 30
DEFAULT_RECORDING_SAMPLE_RATE = 44_100


@dataclass(slots=True)
class RecordingSession:
    target: Path
    sample_rate: int
    max_duration_seconds: int
    stop_event: threading.Event = field(default_factory=threading.Event)
    finished_event: threading.Event = field(default_factory=threading.Event)
    frames: list[np.ndarray] = field(default_factory=list)
    frame_count: int = 0
    thread: threading.Thread | None = None
    error: str | None = None
    reached_limit: bool = False


def build_notebook_app(analyzer: ScaleAnalyzer | None = None, captures_dir: str | Path = "notebooks/_captures") -> VBox:
    analyzer = analyzer or ScaleAnalyzer()
    captures_path = Path(captures_dir)
    captures_path.mkdir(parents=True, exist_ok=True)
    recording_session: RecordingSession | None = None
    culture_list = ", ".join(analyzer.supported_cultures())
    template_count = len(analyzer.templates)

    upload = FileUpload(accept=".wav,.flac,.mp3,.ogg,.m4a,.aac", multiple=False, description="Choose file")
    analyze_button = Button(description="Analyze uploaded file", button_style="primary", icon="upload")
    start_record_button = Button(description="Start recording", button_style="success", icon="microphone")
    stop_record_button = Button(description="Stop recording", button_style="danger", icon="stop", disabled=True)
    section_intro = WidgetHTML(
        (
            "<div style='padding:12px 14px;background:#f4f7fb;border:1px solid #d9e2ec;border-radius:10px;"
            "margin-bottom:12px;'>"
            "<h3 style='margin:0 0 8px 0;'>Culture Scale Lab</h3>"
            "<div style='color:#4f5b67;'>Upload a file or record from the microphone. "
            "Recording uses the input device's preferred sample rate when available and auto-stops at "
            f"{MAX_RECORDING_SECONDS} seconds for safety.</div>"
            f"<div style='color:#5a6472;font-size:13px;margin-top:8px;'>"
            f"Starter library: {template_count} templates across {culture_list}. "
            "Results are closest supported matches, not guaranteed complete world-coverage identifications."
            "</div>"
            "</div>"
        )
    )
    upload_section = WidgetHTML(
        (
            "<div style='margin:8px 0 6px 0;font-weight:600;'>Upload Audio File</div>"
            "<div style='color:#5a6472;font-size:13px;margin-bottom:4px;'>"
            "Use this when you already have a recording to analyze."
            "</div>"
        )
    )
    record_section = WidgetHTML(
        (
            "<div style='margin:12px 0 6px 0;font-weight:600;'>Live Microphone Recording</div>"
            "<div style='color:#5a6472;font-size:13px;margin-bottom:4px;'>"
            "Press start, perform or sing, then press stop when you're done."
            "</div>"
        )
    )
    status = WidgetHTML("<b>Status:</b> Ready.")
    output = Output()

    def set_status(message: str) -> None:
        status.value = f"<b>Status:</b> {message}"

    def set_recording_controls(*, recording: bool) -> None:
        start_record_button.disabled = recording
        stop_record_button.disabled = not recording

    def analyze_path(path: Path, source_label: str, *, delete_after_analysis: bool = False) -> None:
        set_status(f"Analyzing {source_label}...")
        try:
            with output:
                output.clear_output(wait=True)
                plt.close("all")
                result = analyzer.analyze_file(path)
                display(Markdown(result.summary_markdown()))
                display(Audio(filename=str(path)))
                display(render_match_table(result, limit=8))
                figure = plot_result(result)
                display(figure)
                plt.close(figure)
            set_status(f"Finished analyzing {source_label}.")
        finally:
            if delete_after_analysis:
                try:
                    path.unlink(missing_ok=True)
                except Exception as exc:
                    set_status(f"Finished analyzing {source_label}, but temporary live recording cleanup failed: {exc}")

    def on_analyze_clicked(_: Button) -> None:
        uploaded = first_uploaded_file(upload.value)
        if uploaded is None:
            set_status("Upload a file first.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = captures_path / f"{timestamp}_{uploaded['name']}"
        target.write_bytes(uploaded["content"])
        analyze_path(target, uploaded["name"])

    def on_start_recording(_: Button) -> None:
        nonlocal recording_session
        if recording_session is not None and not recording_session.finished_event.is_set():
            set_status("A recording is already in progress.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = captures_path / f"mic_capture_{timestamp}.wav"
        sample_rate = best_recording_sample_rate()
        recording_session = RecordingSession(
            target=target,
            sample_rate=sample_rate,
            max_duration_seconds=MAX_RECORDING_SECONDS,
        )
        recording_session.thread = threading.Thread(
            target=record_microphone_background,
            args=(recording_session,),
            daemon=True,
        )
        recording_session.thread.start()
        set_recording_controls(recording=True)
        set_status(
            "Recording from the microphone at "
            f"{sample_rate:,} Hz. Press Stop when finished. "
            f"Auto-stop at {MAX_RECORDING_SECONDS} seconds."
        )

    def on_stop_recording(_: Button) -> None:
        nonlocal recording_session
        if recording_session is None:
            set_status("There is no active recording to stop.")
            return

        session = recording_session
        set_status("Stopping recording...")
        session.stop_event.set()
        if session.thread is not None:
            session.thread.join(timeout=5.0)

        if not session.finished_event.is_set():
            set_status("The recorder is still shutting down. Press Stop again in a moment if needed.")
            return

        recording_session = None
        set_recording_controls(recording=False)

        if session.error is not None:
            set_status(f"Microphone capture failed: {session.error}")
            return

        source_label = "microphone capture"
        if session.reached_limit:
            source_label += f" (auto-stopped at {session.max_duration_seconds}s)"
        analyze_path(session.target, source_label, delete_after_analysis=True)

    analyze_button.on_click(on_analyze_clicked)
    start_record_button.on_click(on_start_recording)
    stop_record_button.on_click(on_stop_recording)

    controls = VBox(
        [
            section_intro,
            upload_section,
            HBox([upload, analyze_button]),
            record_section,
            HBox([start_record_button, stop_record_button]),
            status,
            output,
        ]
    )
    return controls


def first_uploaded_file(value: object) -> dict[str, object] | None:
    if not value:
        return None

    if isinstance(value, dict):
        first = next(iter(value.values()))
        return {
            "name": first["name"],
            "content": bytes(first["content"]),
        }

    first = value[0]
    return {
        "name": first["name"],
        "content": bytes(first["content"]),
    }


def choose_recording_sample_rate(default_samplerate: float | int | None) -> int:
    if default_samplerate is None:
        return DEFAULT_RECORDING_SAMPLE_RATE

    try:
        sample_rate = int(round(float(default_samplerate)))
    except (TypeError, ValueError):
        return DEFAULT_RECORDING_SAMPLE_RATE

    if 16_000 <= sample_rate <= 96_000:
        return sample_rate

    common_rates = (48_000, 44_100, 32_000, 22_050, 16_000)
    return min(common_rates, key=lambda rate: abs(rate - sample_rate))


def best_recording_sample_rate() -> int:
    import sounddevice as sd

    try:
        input_device = sd.query_devices(kind="input")
    except Exception:  # pragma: no cover - hardware dependent
        return DEFAULT_RECORDING_SAMPLE_RATE

    default_samplerate = None
    if isinstance(input_device, dict):
        default_samplerate = input_device.get("default_samplerate")
    return choose_recording_sample_rate(default_samplerate)


def record_microphone_background(session: RecordingSession) -> None:
    import sounddevice as sd

    max_frames = session.sample_rate * session.max_duration_seconds

    try:
        def callback(indata, frames, _time, status) -> None:
            if status and session.error is None:
                session.error = str(status)

            remaining = max_frames - session.frame_count
            if remaining <= 0:
                session.reached_limit = True
                session.stop_event.set()
                raise sd.CallbackStop()

            chunk = np.array(indata[: min(frames, remaining), 0], copy=True)
            if chunk.size > 0:
                session.frames.append(chunk)
                session.frame_count += int(chunk.size)

            if session.frame_count >= max_frames:
                session.reached_limit = True
                session.stop_event.set()
                raise sd.CallbackStop()

            if session.stop_event.is_set():
                raise sd.CallbackStop()

        with sd.InputStream(
            samplerate=session.sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            while not session.stop_event.wait(0.1):
                pass
    except Exception as exc:  # pragma: no cover - hardware dependent
        session.error = str(exc)
    finally:
        try:
            if session.frames:
                audio = np.concatenate(session.frames).astype(np.float32)
                sf.write(session.target, audio, session.sample_rate)
            elif session.error is None:
                session.error = "No audio was captured."
        except Exception as exc:  # pragma: no cover - hardware dependent
            session.error = f"Failed to save the recording: {exc}"
        session.finished_event.set()


def plot_result(result: AnalysisResult):
    features = result.features
    if features is None:
        raise ValueError("Feature plots require an analysis result with extracted features.")

    best = result.best_match
    figure = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.3))
    ax_matches = figure.add_subplot(grid[0, 0])
    ax_cultures = figure.add_subplot(grid[0, 1])
    ax_chroma = figure.add_subplot(grid[1, 0])
    ax_template = figure.add_subplot(grid[1, 1])

    top_matches = result.matches[:5]
    top_labels = [short_match_label(match) for match in top_matches][::-1]
    top_scores = [match.score for match in top_matches][::-1]
    top_colors = ["#9db4c0"] * len(top_matches)
    if top_colors:
        top_colors[-1] = "#2a6f97"
    ax_matches.barh(top_labels, top_scores, color=top_colors)
    ax_matches.set_xlim(0.0, 1.0)
    ax_matches.set_title("Top Candidate Matches")
    ax_matches.set_xlabel("Score")
    ax_matches.grid(axis="x", alpha=0.25, linestyle="--")

    cultures = list(result.culture_scores.keys())
    culture_scores = list(result.culture_scores.values())
    ax_cultures.bar(cultures, culture_scores, color=["#d62828", "#f4a261", "#457b9d"][: len(cultures)])
    ax_cultures.set_ylim(0.0, 1.0)
    ax_cultures.set_title("Culture Score Breakdown")
    ax_cultures.set_ylabel("Normalized score")
    ax_cultures.grid(axis="y", alpha=0.25, linestyle="--")

    chroma_positions = np.arange(12)
    ax_chroma.bar(chroma_positions, features.combined_12, color="#3a7ca5")
    ax_chroma.set_title("Observed 12-Tone Pitch Emphasis")
    ax_chroma.set_xticks(chroma_positions)
    ax_chroma.set_xticklabels(NOTE_NAMES_12)
    ax_chroma.set_ylim(0, max(0.18, float(features.combined_12.max()) * 1.3))
    ax_chroma.set_ylabel("Relative energy")
    ax_chroma.grid(axis="y", alpha=0.25, linestyle="--")

    top_note_indices = np.argsort(features.combined_12)[-3:]
    for index in sorted(top_note_indices):
        ax_chroma.text(
            index,
            float(features.combined_12[index]) + 0.01,
            NOTE_NAMES_12[index],
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1d3557",
        )

    observed_profile, overlay, x_positions, tick_positions, tick_labels, title = template_plot_data(result)
    ax_template.bar(x_positions, observed_profile, color="#e07a5f", alpha=0.85, label="Observed")
    ax_template.plot(x_positions, overlay, color="#6a040f", linewidth=2.5, label="Best template")
    ax_template.set_title(title)
    ax_template.set_xticks(tick_positions)
    ax_template.set_xticklabels(tick_labels, rotation=0)
    ax_template.set_ylim(0, max(0.14, float(max(np.max(observed_profile), np.max(overlay))) * 1.3))
    ax_template.set_ylabel("Relative weight")
    ax_template.grid(axis="y", alpha=0.25, linestyle="--")
    ax_template.legend(frameon=False, loc="upper right")

    summary = (
        f"Best match: {best.template.culture} {best.template.name} | "
        f"Tonic: {best.tonic_label} | "
        f"Score: {best.score:.3f} | "
        f"Voiced frames: {features.voiced_frame_count}"
    )
    figure.suptitle(summary, fontsize=15, fontweight="semibold")
    return figure


def short_match_label(match) -> str:
    return f"{match.template.culture} | {match.template.name} | {match.tonic_label}"


def template_plot_data(result: AnalysisResult) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], str]:
    features = result.features
    if features is None:
        raise ValueError("Template plot data requires extracted features.")

    best = result.best_match
    bins = best.template.bins
    active = np.array([(best.tonic_bin + interval) % bins for interval in best.template.intervals], dtype=int)
    overlay = build_template_vector(active, bins)

    if bins == 24:
        observed = features.pitch_class_24
        if not np.any(observed):
            observed = np.zeros(24, dtype=float)
            observed[::2] = features.combined_12
            observed = observed / observed.sum() if observed.sum() > 0 else observed
        x_positions = np.arange(24)
        tick_positions = np.arange(0, 24, 2)
        tick_labels = [note_name_for_bin(index, 24) for index in tick_positions]
        title = "Best Template Fit in Quarter-Tone Space"
        return observed, overlay, x_positions, tick_positions, tick_labels, title

    observed = features.combined_12
    x_positions = np.arange(12)
    tick_positions = np.arange(12)
    tick_labels = list(NOTE_NAMES_12)
    title = "Best Template Fit in 12-Tone Space"
    return observed, overlay, x_positions, tick_positions, tick_labels, title


def render_match_table(result: AnalysisResult, limit: int = 8) -> DisplayHTML:
    rows_html: list[str] = []
    for match in result.matches[:limit]:
        example = lookup_example(match.template)
        example_html = "<span class='mc-muted'>No curated listening example yet.</span>"
        actions_html = "<span class='mc-muted'>No links available.</span>"

        if example is not None:
            example_html = (
                f"<strong>{html.escape(example.title)}</strong>"
                f"<div class='mc-muted'>{html.escape(example.artist)}</div>"
                f"<div class='mc-small'>{html.escape(example.description)}</div>"
            )
            actions = [
                (
                    example.youtube_url,
                    "Play / search on YouTube",
                ),
                (
                    example.spotify_url,
                    "Open on Spotify",
                ),
            ]
            actions_html = " ".join(
                (
                    f"<a class='mc-link' href='{html.escape(url, quote=True)}' "
                    f"target='_blank' rel='noopener noreferrer'>{html.escape(label)}</a>"
                )
                for url, label in actions
            )

        rows_html.append(
            (
                "<tr>"
                f"<td>{html.escape(match.template.culture)}</td>"
                f"<td>{html.escape(match.template.family)}</td>"
                f"<td>{html.escape(match.template.name)}</td>"
                f"<td>{html.escape(match.tonic_label)}</td>"
                f"<td>{match.score:.3f}</td>"
                f"<td>{example_html}</td>"
                f"<td>{actions_html}</td>"
                "</tr>"
            )
        )

    table_html = (
        "<style>"
        ".mc-table{border-collapse:collapse;width:100%;font-size:14px;margin-top:12px;}"
        ".mc-table th,.mc-table td{border:1px solid #d9d9d9;padding:10px;vertical-align:top;text-align:left;}"
        ".mc-table th{background:#f4f7fb;font-weight:600;}"
        ".mc-muted{color:#5a6472;}"
        ".mc-small{font-size:12px;color:#5a6472;margin-top:4px;}"
        ".mc-link{display:inline-block;margin-right:10px;margin-bottom:6px;}"
        "</style>"
        "<h4>Top Suggestions With Representative Listening Examples</h4>"
        "<p class='mc-small'>Examples are curated listening pointers for the predicted family or scale area. "
        "For traditions such as dastgah and thaat, they are illustrative references rather than definitive song-to-scale ground truth.</p>"
        "<table class='mc-table'>"
        "<thead><tr>"
        "<th>Culture</th><th>Family</th><th>Scale</th><th>Tonic</th><th>Score</th>"
        "<th>Representative Example</th><th>Actions</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>"
    )
    return DisplayHTML(table_html)

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

NOTE_NAMES_12 = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


@dataclass(frozen=True, slots=True)
class ScaleTemplate:
    culture: str
    family: str
    name: str
    bins: int
    intervals: tuple[int, ...]
    description: str
    citation_hint: str

    @property
    def label(self) -> str:
        return f"{self.culture} {self.family}: {self.name}"


@lru_cache(maxsize=1)
def build_default_templates() -> list[ScaleTemplate]:
    data_path = Path(__file__).resolve().parent / "data" / "scale_templates.json"
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    templates: list[ScaleTemplate] = []
    for item in payload:
        templates.append(
            ScaleTemplate(
                culture=str(item["culture"]),
                family=str(item["family"]),
                name=str(item["name"]),
                bins=int(item["bins"]),
                intervals=tuple(int(value) for value in item["intervals"]),
                description=str(item["description"]),
                citation_hint=str(item["citation_hint"]),
            )
        )
    return templates


def supported_cultures() -> list[str]:
    return sorted({template.culture for template in build_default_templates()})


def note_name_for_bin(bin_index: int, bins: int) -> str:
    if bins == 12:
        return NOTE_NAMES_12[bin_index % 12]

    semitone = (bin_index // 2) % 12
    if bin_index % 2 == 0:
        return NOTE_NAMES_12[semitone]
    return f"{NOTE_NAMES_12[semitone]} +50c"

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote_plus

from .catalog import ScaleTemplate


@dataclass(frozen=True, slots=True)
class ListeningExample:
    title: str
    artist: str
    description: str
    youtube_query: str
    spotify_query: str | None = None
    youtube_video_id: str | None = None

    @property
    def youtube_url(self) -> str:
        if self.youtube_video_id:
            return f"https://www.youtube.com/watch?v={quote_plus(self.youtube_video_id)}"
        return f"https://www.youtube.com/results?search_query={quote_plus(self.youtube_query)}"

    @property
    def spotify_url(self) -> str:
        query = self.spotify_query or f"{self.title} {self.artist}"
        return f"https://open.spotify.com/search/{quote_plus(query)}"

    @property
    def embed_url(self) -> str | None:
        if not self.youtube_video_id:
            return None
        return f"https://www.youtube.com/embed/{quote_plus(self.youtube_video_id)}"


TEMPLATE_EXAMPLES: dict[tuple[str, str, str], ListeningExample] = {
    (
        "Persian",
        "Dastgah",
        "Shur / Nava",
    ): ListeningExample(
        title="Nava",
        artist="Mohammad Reza Shajarian",
        description="Representative Persian art-music listening pointer for the Nava/Shur modal area.",
        youtube_query="Mohammad Reza Shajarian Nava dastgah",
    ),
    (
        "Persian",
        "Dastgah",
        "Segah",
    ): ListeningExample(
        title="Dastgah-e Segah",
        artist="Parisa",
        description="Representative classical recording often used as a Segah listening reference.",
        youtube_query="Parisa Dastgah Segah",
    ),
    (
        "Persian",
        "Dastgah",
        "Chahargah",
    ): ListeningExample(
        title="Chahargah",
        artist="Shahram Nazeri",
        description="Representative Chahargah performance reference.",
        youtube_query="Shahram Nazeri Chahargah",
    ),
    (
        "Persian",
        "Dastgah",
        "Homayun",
    ): ListeningExample(
        title="Bidad",
        artist="Mohammad Reza Shajarian",
        description="Well-known Homayun-area listening reference from the Persian vocal tradition.",
        youtube_query="Mohammad Reza Shajarian Bidad Homayun",
    ),
    (
        "Persian",
        "Dastgah",
        "Mahur / Rast-Panjgah",
    ): ListeningExample(
        title="Morghe Sahar",
        artist="Mohammad Reza Shajarian",
        description="A widely recognized Persian song often used as a Mahur-family listening pointer.",
        youtube_query="Mohammad Reza Shajarian Morghe Sahar",
    ),
    (
        "Indian",
        "Thaat",
        "Bilawal",
    ): ListeningExample(
        title="Raga Alhaiya Bilawal",
        artist="Pandit Jasraj",
        description="Representative Bilawal-thaat listening example through a classic raga performance.",
        youtube_query="Pandit Jasraj Raga Alhaiya Bilawal",
    ),
    (
        "Indian",
        "Thaat",
        "Kalyan",
    ): ListeningExample(
        title="Raga Yaman",
        artist="Kishori Amonkar",
        description="Classic Kalyan-thaat listening example.",
        youtube_query="Kishori Amonkar Raga Yaman",
    ),
    (
        "Indian",
        "Thaat",
        "Khamaj",
    ): ListeningExample(
        title="Raga Khamaj",
        artist="Rashid Khan",
        description="Representative Khamaj-thaat listening example.",
        youtube_query="Rashid Khan Raga Khamaj",
    ),
    (
        "Indian",
        "Thaat",
        "Kafi",
    ): ListeningExample(
        title="Raga Kafi",
        artist="Veena Sahasrabuddhe",
        description="Representative Kafi-thaat listening example.",
        youtube_query="Veena Sahasrabuddhe Raga Kafi",
    ),
    (
        "Indian",
        "Thaat",
        "Asavari",
    ): ListeningExample(
        title="Raga Asavari",
        artist="Kishori Amonkar",
        description="Representative Asavari-thaat listening example.",
        youtube_query="Kishori Amonkar Raga Asavari",
    ),
    (
        "Indian",
        "Thaat",
        "Bhairavi",
    ): ListeningExample(
        title="Raga Bhairavi",
        artist="Pandit Bhimsen Joshi",
        description="Representative Bhairavi-thaat listening example.",
        youtube_query="Bhimsen Joshi Raga Bhairavi",
    ),
    (
        "Indian",
        "Thaat",
        "Bhairav",
    ): ListeningExample(
        title="Raga Bhairav",
        artist="Ustad Bismillah Khan",
        description="Representative Bhairav-thaat listening example.",
        youtube_query="Bismillah Khan Raga Bhairav",
    ),
    (
        "Indian",
        "Thaat",
        "Poorvi",
    ): ListeningExample(
        title="Raga Poorvi",
        artist="Pandit Jasraj",
        description="Representative Poorvi-thaat listening example.",
        youtube_query="Pandit Jasraj Raga Poorvi",
    ),
    (
        "Indian",
        "Thaat",
        "Marwa",
    ): ListeningExample(
        title="Raga Marwa",
        artist="Kishori Amonkar",
        description="Representative Marwa-thaat listening example.",
        youtube_query="Kishori Amonkar Raga Marwa",
    ),
    (
        "Indian",
        "Thaat",
        "Todi",
    ): ListeningExample(
        title="Miyan Ki Todi",
        artist="Pandit Bhimsen Joshi",
        description="Representative Todi-thaat listening example.",
        youtube_query="Bhimsen Joshi Miyan Ki Todi",
    ),
    (
        "Chinese",
        "Pentatonic Mode",
        "Gong",
    ): ListeningExample(
        title="Mo Li Hua (Jasmine Flower)",
        artist="Traditional",
        description="Illustrative Gong-mode listening pointer from a widely recognized Chinese melody.",
        youtube_query="Mo Li Hua Jasmine Flower traditional Chinese song",
    ),
    (
        "Chinese",
        "Pentatonic Mode",
        "Shang",
    ): ListeningExample(
        title="Kangding Love Song",
        artist="Traditional",
        description="Illustrative Shang-mode listening pointer.",
        youtube_query="Kangding Love Song traditional Chinese",
    ),
    (
        "Chinese",
        "Pentatonic Mode",
        "Jue",
    ): ListeningExample(
        title="Fengyang Flower Drum",
        artist="Traditional",
        description="Illustrative Jue-mode listening pointer.",
        youtube_query="Fengyang Flower Drum traditional Chinese",
    ),
    (
        "Chinese",
        "Pentatonic Mode",
        "Zhi",
    ): ListeningExample(
        title="Fisherman's Song at Dusk",
        artist="Traditional",
        description="Illustrative Zhi-mode listening pointer.",
        youtube_query="Fisherman's Song at Dusk traditional Chinese",
    ),
    (
        "Chinese",
        "Pentatonic Mode",
        "Yu",
    ): ListeningExample(
        title="Phoenix-Tail Bamboo Under the Moonlight",
        artist="Traditional",
        description="Illustrative Yu-mode listening pointer.",
        youtube_query="Phoenix Tail Bamboo Under the Moonlight traditional Chinese",
    ),
}


CULTURE_FALLBACKS: dict[str, ListeningExample] = {
    "Persian": ListeningExample(
        title="Persian Dastgah Sampler",
        artist="Traditional",
        description="General Persian dastgah listening fallback.",
        youtube_query="Persian dastgah traditional music",
    ),
    "Indian": ListeningExample(
        title="Hindustani Raga Sampler",
        artist="Traditional",
        description="General Hindustani listening fallback.",
        youtube_query="Hindustani classical raga performance",
    ),
    "Chinese": ListeningExample(
        title="Chinese Pentatonic Sampler",
        artist="Traditional",
        description="General Chinese pentatonic listening fallback.",
        youtube_query="traditional Chinese pentatonic music",
    ),
    "Arabic": ListeningExample(
        title="Arabic Maqam Sampler",
        artist="Traditional",
        description="General Arabic maqam listening fallback.",
        youtube_query="Arabic maqam traditional music",
    ),
    "Turkish": ListeningExample(
        title="Turkish Makam Sampler",
        artist="Traditional",
        description="General Turkish makam listening fallback.",
        youtube_query="Turkish makam traditional music",
    ),
    "Japanese": ListeningExample(
        title="Japanese Pentatonic Sampler",
        artist="Traditional",
        description="General Japanese pentatonic listening fallback.",
        youtube_query="traditional Japanese pentatonic music",
    ),
    "Western": ListeningExample(
        title="Western Modal Sampler",
        artist="Traditional / Classical",
        description="General Western modal listening fallback.",
        youtube_query="Western modal music examples",
    ),
}


def lookup_example(template: ScaleTemplate) -> ListeningExample | None:
    return TEMPLATE_EXAMPLES.get(
        (template.culture, template.family, template.name),
        CULTURE_FALLBACKS.get(template.culture),
    )

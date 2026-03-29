from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyzer import ScaleAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate musical scale culture and tonic from audio.")
    parser.add_argument("path", type=Path, help="Path to the audio file to analyze.")
    parser.add_argument("--top-k", type=int, default=8, help="How many candidate matches to return.")
    args = parser.parse_args()

    analyzer = ScaleAnalyzer()
    result = analyzer.analyze_file(args.path, top_k=args.top_k)
    payload = {
        "source_path": str(args.path),
        "best_match": {
            "culture": result.best_match.template.culture,
            "family": result.best_match.template.family,
            "scale": result.best_match.template.name,
            "tonic": result.best_match.tonic_label,
            "score": round(result.best_match.score, 4),
        },
        "culture_scores": {key: round(value, 4) for key, value in result.culture_scores.items()},
        "matches": result.as_dataframe(limit=args.top_k).to_dict(orient="records"),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()


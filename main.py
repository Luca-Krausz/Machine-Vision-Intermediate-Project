"""CLI entry point for the seedling morphological analysis pipeline.

Usage
-----
    python main.py <image_path> [--px-per-mm SCALE] [--output results.json]

Examples
--------
    # Basic analysis (pixel units only)
    python main.py sample.jpg

    # With physical scale (10 px = 1 mm) and JSON output
    python main.py sample.jpg --px-per-mm 10 --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys

from src.pipeline import analyse_seedling


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Automated morphological analysis of eucalyptus/pine seedlings. "
            "Extracts plant height, collar diameter, leaf area and leaf count "
            "from a seedling image captured against a blue background."
        )
    )
    parser.add_argument("image", help="Path to the input seedling image.")
    parser.add_argument(
        "--px-per-mm",
        type=float,
        default=None,
        metavar="SCALE",
        help=(
            "Spatial scale in pixels per millimetre.  When provided, the "
            "output includes physical measurements (mm / mm²) in addition to "
            "pixel measurements."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Optional path to save results as a JSON file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        results = analyse_seedling(args.image, px_per_mm=args.px_per_mm)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output = json.dumps(results, indent=2)
    print(output)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(output)
        print(f"\nResults saved to {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

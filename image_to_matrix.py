"""
CLI utility to convert an image into a matrix of values scaled between 0 and 256.

The script loads an image, converts it to grayscale, rescales the intensity range
so the minimum pixel maps to 0 and the maximum pixel maps to 256, and then emits
the resulting matrix either to stdout or to a file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, Iterable, List

from matrix_utils import (
    format_matrix_as_csv,
    format_matrix_as_json,
    format_matrix_as_python,
    image_to_matrix,
)


Matrix = List[List[int]]
FORMATTERS: Dict[str, Callable[[Matrix], str]] = {
    "python": format_matrix_as_python,
    "json": format_matrix_as_json,
    "csv": format_matrix_as_csv,
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an image into a matrix scaled between 0 and 256."
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional path to write the matrix. Defaults to stdout when omitted.",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=("python", "json", "csv"),
        default="python",
        help="Output format for the matrix (default: python).",
    )
    parser.add_argument(
        "--no-rescale",
        dest="rescale",
        action="store_false",
        help="Skip rescaling and emit raw grayscale values (0-255).",
    )
    parser.set_defaults(rescale=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.image.exists():
        raise SystemExit(f"Image does not exist: {args.image}")

    matrix = image_to_matrix(args.image, rescale=args.rescale)
    output = FORMATTERS[args.format](matrix)

    if args.output:
        args.output.write_text(output)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

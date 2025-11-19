"""
Helper utilities for converting images into numeric matrices.
"""

from __future__ import annotations

import ast
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union, TYPE_CHECKING

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt

if TYPE_CHECKING:  
    from werkzeug.datastructures import FileStorage

Matrix = List[List[int]]
ImageSource = Union[str, Path, bytes, "Image.Image", "FileLike"]


def _open_image(source: ImageSource) -> Image.Image:
    """
    Open an image from the supported source types.

    `source` can be one of:
      - str or pathlib.Path pointing to an image on disk
      - bytes containing the encoded image
      - PIL Image instance (returned as-is)
      - file-like object providing a `.read()` method
    """
    if isinstance(source, Image.Image):
        return source

    if isinstance(source, (str, Path)):
        return Image.open(source)

    if isinstance(source, bytes):
        from io import BytesIO

        return Image.open(BytesIO(source))

    seek = getattr(source, "seek", None)
    if callable(seek):
        seek(0)
    return Image.open(source)


def image_to_matrix(
    source: ImageSource,
    size: Tuple[int, int] | None = None,
    rescale: bool = True,
) -> Matrix:
    with _open_image(source) as img:
        grayscale = img.convert("L")
        if size:
            grayscale = grayscale.resize(size, Image.LANCZOS)
        width, height = grayscale.size
        pixels = list(grayscale.getdata())

    matrix: Matrix = [pixels[row * width : (row + 1) * width] for row in range(height)]

    return rescale_matrix(matrix) if rescale else matrix


def rescale_matrix(matrix: Matrix) -> Matrix:
    flat = [value for row in matrix for value in row]
    min_value = min(flat)
    max_value = max(flat)

    if min_value == max_value:
        return [[0 for _ in row] for row in matrix]

    scale = 256 / (max_value - min_value)
    rescaled: Matrix = []
    for row in matrix:
        rescaled_row: List[int] = []
        for value in row:
            scaled_value = round((value - min_value) * scale)
            rescaled_row.append(min(256, max(0, int(scaled_value))))
        rescaled.append(rescaled_row)

    return rescaled


def format_matrix_as_python(matrix: Matrix) -> str:
    rows = ",\n".join(f"  {row}" for row in matrix)
    return "[\n" + rows + "\n]"


def format_matrix_as_json(matrix: Matrix) -> str:
    import json

    return json.dumps(matrix)


def format_matrix_as_csv(matrix: Matrix) -> str:
    return "\n".join(",".join(str(value) for value in row) for row in matrix)


def format_matrix_as_sage(matrix: Matrix, variable_name: str = "M") -> str:
    lines = [f"{variable_name} = matrix(["]
    for row in matrix:
        row_str = ", ".join(f"{value:3d}" for value in row)
        lines.append(f"    [{row_str}],")
    lines.append("])")
    return "\n".join(lines)


def parse_matrix_literal(raw_text: str) -> Matrix:
    """Parse a Sage-style `matrix([...])` block or plain list literal."""

    cleaned = raw_text.strip()
    if not cleaned:
        raise ValueError("Matrix text is empty.")

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate the numeric matrix inside the text.")

    literal = cleaned[start : end + 1]
    try:
        parsed = ast.literal_eval(literal)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("Matrix text is not valid Python list syntax.") from exc

    if not isinstance(parsed, (list, tuple)) or not parsed:
        raise ValueError("Matrix must be a non-empty list of rows.")

    matrix: Matrix = []
    expected_length: int | None = None
    for row in parsed:
        if not isinstance(row, (list, tuple)):
            raise ValueError("Each row must be expressed as a list of values.")

        if expected_length is None:
            expected_length = len(row)
            if expected_length == 0:
                raise ValueError("Rows cannot be empty.")
        elif len(row) != expected_length:
            raise ValueError("All rows must have the same number of entries.")

        converted_row: List[int] = []
        for value in row:
            if isinstance(value, bool) or isinstance(value, (int, float)):
                converted_row.append(int(round(float(value))))
            else:
                raise ValueError("Matrix values must be numeric.")

        matrix.append(converted_row)

    return matrix


def matrix_to_plot_png(
    matrix: Matrix,
    *,
    cmap: str = "gray",
    vmin: float | None = 0,
    vmax: float | None = 255,
    title: str | None = None,
    show_colorbar: bool = True,
) -> bytes:
    """Render a matrix as a PNG plot similar to Sage's `matrix_plot`."""

    if not matrix or not matrix[0]:
        raise ValueError("Matrix cannot be empty.")

    array = np.array(matrix, dtype=float)
    computed_vmin = float(array.min() if vmin is None else vmin)
    computed_vmax = float(array.max() if vmax is None else vmax)

    if computed_vmin >= computed_vmax:
        raise ValueError("vmin must be smaller than vmax.")

    height, width = array.shape
    figure_width = max(3.0, width / 10)
    figure_height = max(3.0, height / 10)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    im = ax.imshow(
        array,
        cmap=cmap,
        vmin=computed_vmin,
        vmax=computed_vmax,
        origin="upper",
        aspect="equal",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)

    if show_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout(pad=0.25)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def video_to_matrices(
    upload: "FileStorage",
    size: Tuple[int, int] | None = None,
    rescale: bool = True,
    frame_skip: int = 1,
    max_frames: int = 50,
) -> List[Matrix]:
    if frame_skip < 1:
        raise ValueError("frame_skip must be >= 1")
    if max_frames < 1:
        raise ValueError("max_frames must be >= 1")

    import cv2  # Local import so CLI usage does not require OpenCV.
    from tempfile import NamedTemporaryFile

    suffix = Path(upload.filename or "").suffix or ".mp4"
    upload.stream.seek(0)

    matrices: List[Matrix] = []
    with NamedTemporaryFile(suffix=suffix) as tmp:
        upload.save(tmp.name)
        capture = cv2.VideoCapture(tmp.name)

        if not capture.isOpened():
            capture.release()
            raise ValueError("Unable to read the uploaded video.")

        frame_index = 0
        try:
            while len(matrices) < max_frames:
                success, frame = capture.read()
                if not success:
                    break

                if frame_index % frame_skip == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    matrices.append(
                        image_to_matrix(pil_image, size=size, rescale=rescale)
                    )

                frame_index += 1
        finally:
            capture.release()

    return matrices


class FileLike:
    def read(self, *args, **kwargs):  
        raise NotImplementedError

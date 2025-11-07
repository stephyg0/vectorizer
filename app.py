from __future__ import annotations

from typing import Dict, List, Optional

from flask import Flask, render_template, request

from matrix_utils import (
    format_matrix_as_sage,
    image_to_matrix,
    video_to_matrices,
)

UPLOAD_LIMIT_BYTES = 50 * 1024 * 1024 
TARGET_SIZE = (50, 50)
MAX_FRAMES_DISPLAY = 25

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = UPLOAD_LIMIT_BYTES


@app.route("/", methods=["GET", "POST"])
def index():
    image_output: Optional[str] = None
    image_error: Optional[str] = None
    video_outputs: Optional[List[Dict[str, str]]] = None
    video_error: Optional[str] = None

    if request.method == "POST":
        mode = request.form.get("mode", "image")

        if mode == "image":
            file = request.files.get("image")
            if not file or file.filename == "":
                image_error = "Please choose an image to upload."
            else:
                try:
                    matrix = image_to_matrix(file.stream, size=TARGET_SIZE, rescale=True)
                    image_output = format_matrix_as_sage(matrix, variable_name="M")
                except Exception as exc:  
                    image_error = f"Failed to process image: {exc}"

        elif mode == "video":
            file = request.files.get("video")

            if not file or file.filename == "":
                video_error = "Please choose a video to upload."
            else:
                frame_skip = _safe_int(request.form.get("frame_skip"), default=1, min_value=1)
                max_frames = _safe_int(
                    request.form.get("max_frames"),
                    default=10,
                    min_value=1,
                    max_value=MAX_FRAMES_DISPLAY,
                )

                try:
                    matrices = video_to_matrices(
                        file,
                        size=TARGET_SIZE,
                        rescale=True,
                        frame_skip=frame_skip,
                        max_frames=max_frames,
                    )
                    if not matrices:
                        video_error = "No readable frames were detected in this video."
                    else:
                        video_outputs = [
                            {
                                "label": f"Frame {index}",
                                "matrix": format_matrix_as_sage(
                                    matrix, variable_name=f"M_{index:03d}"
                                ),
                            }
                            for index, matrix in enumerate(matrices, start=1)
                        ]
                except Exception as exc:  
                    video_error = f"Failed to process video: {exc}"

    return render_template(
        "index.html",
        image_output=image_output,
        image_error=image_error,
        video_outputs=video_outputs,
        video_error=video_error,
        max_frames=MAX_FRAMES_DISPLAY,
    )


def _safe_int(
    raw_value: Optional[str],
    *,
    default: int,
    min_value: int,
    max_value: Optional[int] = None,
) -> int:
    try:
        value = int(raw_value) if raw_value is not None else default
    except (TypeError, ValueError):
        value = default

    value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


if __name__ == "__main__":
    port = 8000
    print(f"Starting server on http://127.0.0.1:{port}")
    app.run(debug=True, port=port)

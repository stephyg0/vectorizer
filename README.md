# Image & Video Matrix Converter

This project turns pixel data into Sage-compatible matrices. It ships with:

- A command-line utility for converting any image into a matrix whose values are rescaled between `0` and `256`.
- A Flask web app that handles both single-image uploads and video clips, resizing content to `50×50` before generating matrices such as:

```text
M = matrix([
    [255,255, ...],
    [...],
])
```

## Requirements

- Python 3.8+
- Node.js (only for running the `npm run dev` helper script)
- Install Python dependencies: `pip install -r requirements.txt`

## CLI Workflow

Convert a local image and print the Python-list representation:

```bash
python image_to_matrix.py path/to/image.png
```

Other useful options:

- `--format csv|json|python` chooses the output encoding.
- `--output matrix.txt` writes to a file instead of stdout.
- `--no-rescale` keeps the original `0–255` grayscale values instead of stretching them to `0–256`.

The CLI stops automatically if the image contains no intensity variation (all zeros after scaling).

## Web App Workflow

Start the Flask server (npm simply runs the command for you):

```bash
pip install -r requirements.txt
npm run dev   # serves http://127.0.0.1:8000
```

Once opened in the browser you can choose between two panels:

### 1. Image Matrix Converter

- Upload any still image.
- The server converts it to grayscale, resizes it to `50×50`, rescales values to `0–256`, and renders a Sage-friendly `M = matrix([...])` block you can copy directly.

### 2. Video Matrix Converter

- Upload a short video (≤50 MB by default).
- Configure **Frame Skip** to sample every _n_-th frame and **Max Frames** (up to 25) to limit how many matrices you get back.
- Each selected frame is resized, rescaled, and labeled as `M_001`, `M_002`, etc., shown inside collapsible panels so you can expand only the frames you care about.

> Tip: For long clips, increase **Frame Skip** to keep processing quick and results readable.

## Files of Interest

- `matrix_utils.py` — reusable helpers for image resizing, scaling, and video frame extraction.
- `image_to_matrix.py` — CLI entry point built on top of the helpers.
- `app.py`, `templates/index.html`, `static/styles.css` — Flask app, UI template, and styling for the combined image/video experience.

With these pieces you can script matrix exports, or use the browser UI to explore videos frame by frame. Have fun turning pixels into math.

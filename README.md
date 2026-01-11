# EfficientNetB0 Flask Inference API

This is a minimal Flask app that loads your trained EfficientNetB0 `.h5` model and exposes a `/predict` endpoint for image classification.

## Setup

1. Place your model file:
   - Put your `.h5` model at `web/model/model.h5` (or set `MODEL_PATH` in `.env`).
   - Optional: add `web/model/labels.txt` with one label per line to map outputs.

2. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
   - TensorFlow for macOS depends on your CPU:

```bash
# Apple Silicon (M1/M2/M3)
pip install tensorflow-macos
# Intel macs
pip install tensorflow
```

```bash
# Common deps
pip install -r requirements.txt
```

## Run the server

```bash
# Optionally configure environment
cp .env.example .env
# Edit .env if needed

python app.py
# Server listens on http://localhost:5000
# UI is available at http://localhost:5000/ui
```

## Health check

```bash
curl http://localhost:5000/health
```

## Predict

Send a multipart image under field `file`:

```bash
curl -X POST \
  -F "file=@/path/to/image.jpg" \
  http://localhost:5000/predict
```

Response example (with labels):

```json
{
  "predictions": [0.01, 0.05, 0.94],
  "labels": ["cat", "dog", "bird"],
  "top": { "index": 2, "label": "bird", "score": 0.94 }
}
```

If `labels.txt` is not provided, `labels` is omitted and `top` contains only index and score.

## Configuration

- `.env` keys:
  - `PORT`: server port (default `5001`)
  - `IMAGE_SIZE`: input size (default `224` for EfficientNetB0)
  - `MODEL_PATH` (optional): path to a single `.h5` file (legacy). For multi-plant, place files under `model/best_{plant}_model.h5`.

## Notes

- The model is loaded with `compile=False` to avoid requiring training-only objects.
- Images are resized to `IMAGE_SIZE` and preprocessed with EfficientNet's `preprocess_input` (fallback to `[0,1]`).
- If TensorFlow is not installed, `/health` will report an error and `/predict` will return 500 until it is.

---

## Multi-Plant Mode (Grape / Tomato / Potato)

This app supports selecting the plant at request time.

- Place models:
  - `model/best_grape_model.h5`
  - `model/best_tomato_model.h5`
  - `model/best_potato_model.h5`
- Optional labels files:
  - `model/{plant}_labels.txt` or `model/labels.txt`
- UI: `/ui` has a dropdown to select `grape`, `tomato`, or `potato` and uploads the image.
- API: send `plant` field with the image to `/predict`.

Example (macOS/Linux):
```bash
curl -X POST \
  -F "plant=grape" \
  -F "file=@/path/to/image.jpg" \
  http://127.0.0.1:5001/predict
```

Health shows per-plant status:
```bash
curl -s http://127.0.0.1:5001/health | jq
```

## Windows Run (PowerShell)

```powershell
cd C:\Users\<you>\Desktop\Proto\web
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

## EfficientNetB0 Overview

EfficientNetB0 is a compact CNN baseline built via compound scaling of width, depth, and resolution.
- Input size: 224×224×3
- Preprocessing: EfficientNet `preprocess_input`
- Output: softmax probabilities over the model’s classes

# CattleVision-AI Localhost

## Run on Windows

1. Install Python 3.11 (64-bit).
2. Open PowerShell inside this folder.
3. Run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Run on macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` if the browser does not open automatically.

## Correct image

- Upload JPG, JPEG, or PNG.
- Show one cow only.
- Keep the full cow large, clear, and unobstructed.
- A side or rear view similar to the training data works best.

The app always requires YOLO cattle detection. It uses the secondary similarity
gate when a valid full Keras archive is available and automatically falls back
to the included TFLite regression model if the Keras file cannot be opened.

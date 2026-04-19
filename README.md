# DRT Flask App

This converts the original desktop-only DRT tool into a Flask web app.

## What changed

- Web UI for uploading and comparing many datasets.
- Iterative ingest flow: keep adding files one by one or in batches until the stream stops.
- One-click analysis across all loaded datasets.
- Small-multiples Nyquist comparison for large batches.
- DRT overlay plot.
- Button to launch the desktop Tk app from the web UI.

## Files

- `app.py` — Flask web app
- `drt_core.py` — shared numerical core
- `desktop_app.py` — original desktop app
- `templates/index.html` — web interface
- `requirements.txt` — Python dependencies

## Run

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`

## Notes

- The “Open desktop app” action only works if Flask is running on the same local machine with a GUI session.
- The dataset queue is stored in memory for simplicity. Restarting the Flask server clears it.
- This version is aimed at comparing many files at once rather than just a single active file.

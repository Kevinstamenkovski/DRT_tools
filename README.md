# DRT Graph Generator

A full Python desktop app built with **CustomTkinter** for generating **DRT graphs** from electrochemical impedance spectroscopy (EIS) data.

## Features

- Load `.csv` or `.txt` EIS files
- DRTtools-style layout
- Nyquist plot
- Bode magnitude plot
- Bode phase plot
- DRT plot
- Customizable analysis parameters
- Manual lambda or automatic GCV-style lambda selection
- Export DRT CSV
- Export fitted EIS CSV
- Export figure PNG
- Appearance/theme controls
- Demo dataset included inside the app

## Expected input file format

Three columns, no header:

1. `frequency (Hz)`
2. `Zreal (Ohm)`
3. `-Zimag (Ohm)`

Example:

```text
100000 0.21 0.01
50000 0.22 0.03
10000 0.28 0.10
1000 0.45 0.22
100 0.85 0.31
```

## Run

```bash
pip install -r requirements.txt
python app.py
```

## Notes

- The interface is inspired by the DRTtools website and tutorial flow.
- The app implements a practical regularized DRT workflow focused on graph generation and export.
- It does **not** currently implement the full Bayesian DRT or Bayesian Hilbert Transform pipeline.

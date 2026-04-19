import io
import math
import os
import subprocess
import sys
import uuid
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for

from drt_core import fit_drt, gcv_lambda, load_eis_file

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / 'uploads'
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024
app.secret_key = 'dev-secret-change-me'

DATASETS = []


def parse_bool(value, default=False):
    if value is None:
        return default
    return str(value).lower() in {'1', 'true', 'yes', 'on'}


def current_params(form):
    params = {
        'basis': form.get('basis', 'Gaussian'),
        'data_mode': form.get('data_mode', 'Combined Re-Im Data'),
        'inductance_mode': form.get('inductance_mode', 'Fitting with Inductance'),
        'derivative_order': 1 if str(form.get('derivative_order', '2')).startswith('1') else 2,
        'lambda_mode': form.get('lambda_mode', 'Custom'),
        'lambda_value': float(form.get('lambda_value', '0.001')),
        'shape_factor': max(0.05, float(form.get('shape_factor', '0.5'))),
        'n_tau': max(40, int(float(form.get('n_tau', '120')))),
    }
    return params


def preprocess_dataset(freq, z_re, z_im_neg, inductance_mode):
    freq = freq.copy()
    z_re = z_re.copy()
    z_im_neg = z_im_neg.copy()
    if inductance_mode == 'Discard Inductive Data':
        mask = z_im_neg >= 0
        if np.count_nonzero(mask) >= 5:
            freq, z_re, z_im_neg = freq[mask], z_re[mask], z_im_neg[mask]
    include_inductance = inductance_mode == 'Fitting with Inductance'
    return freq, z_re, z_im_neg, include_inductance


def solve_one(dataset, params):
    freq, z_re, z_im_neg, include_inductance = preprocess_dataset(
        dataset['freq'], dataset['z_re'], dataset['z_im_neg'], params['inductance_mode']
    )

    lambda_value = params['lambda_value']
    if params['lambda_mode'] == 'Auto (GCV-style)':
        lambda_value = gcv_lambda(
            freq=freq,
            z_re=z_re,
            z_im_neg=z_im_neg,
            lambdas=np.logspace(-5, 1, 18),
            derivative_order=params['derivative_order'],
            include_inductance=include_inductance,
            data_mode=params['data_mode'],
            basis=params['basis'],
            shape_factor=params['shape_factor'],
            n_tau=params['n_tau'],
        )

    result = fit_drt(
        freq=freq,
        z_re=z_re,
        z_im_neg=z_im_neg,
        n_tau=params['n_tau'],
        lambda_value=lambda_value,
        derivative_order=params['derivative_order'],
        include_inductance=include_inductance,
        data_mode=params['data_mode'],
        basis=params['basis'],
        shape_factor=params['shape_factor'],
        nonnegative=True,
    )

    total_pol = float(np.trapz(result.gamma, np.log(result.tau)))
    return {
        'id': dataset['id'],
        'name': dataset['name'],
        'points': int(len(result.freq)),
        'lambda_value': float(result.lambda_value),
        'rinf': float(result.rinf),
        'inductance': float(result.inductance),
        'residual_norm': float(result.residual_norm),
        'polarization_resistance': total_pol,
        'freq': result.freq,
        'z_re': result.z_re,
        'z_im_neg': result.z_im_neg,
        'z_fit': result.z_fit,
        'tau': result.tau,
        'gamma': result.gamma,
    }


def fig_to_base64(fig):
    import base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def make_overview_plot(results):
    n = len(results)
    cols = min(4, max(1, math.ceil(math.sqrt(n))))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)
    axes = axes.flatten()
    for ax in axes[n:]:
        ax.axis('off')
    for ax, r in zip(axes, results):
        ax.plot(r['z_re'], r['z_im_neg'], 'o', markersize=2)
        ax.plot(np.real(r['z_fit']), -np.imag(r['z_fit']), '-', linewidth=1.2)
        ax.set_title(r['name'][:35], fontsize=9)
        ax.set_xlabel("Z' / Ω")
        ax.set_ylabel("-Z'' / Ω")
        ax.grid(True, alpha=0.25)
    fig.suptitle(f'Nyquist comparison ({n} datasets)', fontsize=14)
    fig.tight_layout()
    return fig_to_base64(fig)


def make_gamma_overlay(results):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for r in results:
        ax.plot(r['tau'], r['gamma'], linewidth=1, alpha=0.8, label=r['name'][:20])
    ax.set_xscale('log')
    ax.set_title('DRT overlay')
    ax.set_xlabel('tau / s')
    ax.set_ylabel('γ(tau) / Ω')
    ax.grid(True, alpha=0.25)
    if len(results) <= 20:
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig_to_base64(fig)


@app.get('/')
def index():
    return render_template('index.html', datasets=DATASETS, results=None, charts=None, params={})


@app.post('/add-dataset')
def add_dataset():
    files = request.files.getlist('files')
    added = 0
    errors = []
    for file in files:
        if not file or not file.filename:
            continue
        suffix = Path(file.filename).suffix.lower() or '.txt'
        temp_name = f"{uuid.uuid4().hex}{suffix}"
        path = UPLOAD_DIR / temp_name
        file.save(path)
        try:
            freq, z_re, z_im_neg = load_eis_file(str(path))
            DATASETS.append({
                'id': uuid.uuid4().hex[:10],
                'name': file.filename,
                'path': str(path),
                'freq': freq,
                'z_re': z_re,
                'z_im_neg': z_im_neg,
            })
            added += 1
        except Exception as exc:
            errors.append(f'{file.filename}: {exc}')
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
    return render_template('index.html', datasets=DATASETS, results=None, charts=None, params={},
                           message=f'Added {added} dataset(s). You can keep adding files one batch at a time until you are ready to compare.',
                           errors=errors)


@app.post('/analyze')
def analyze():
    if not DATASETS:
        return render_template('index.html', datasets=DATASETS, results=None, charts=None, params={},
                               errors=['No datasets loaded yet.'])
    params = current_params(request.form)
    results = [solve_one(dataset, params) for dataset in DATASETS]
    charts = {
        'overview': make_overview_plot(results),
        'gamma_overlay': make_gamma_overlay(results),
    }
    table_rows = []
    for r in results:
        table_rows.append({
            'id': r['id'],
            'name': r['name'],
            'points': r['points'],
            'lambda_value': f"{r['lambda_value']:.6g}",
            'rinf': f"{r['rinf']:.6g}",
            'inductance': f"{r['inductance']:.6g}",
            'residual_norm': f"{r['residual_norm']:.6g}",
            'polarization_resistance': f"{r['polarization_resistance']:.6g}",
        })
    return render_template('index.html', datasets=DATASETS, results=table_rows, charts=charts, params=params,
                           message=f'Processed {len(results)} dataset(s).')


@app.post('/clear')
def clear_all():
    DATASETS.clear()
    return redirect(url_for('index'))


@app.post('/open-desktop')
def open_desktop():
    desktop_path = BASE_DIR / 'desktop_app.py'
    if not desktop_path.exists():
        return render_template('index.html', datasets=DATASETS, results=None, charts=None, params={},
                               errors=['desktop_app.py was not found.'])
    try:
        subprocess.Popen([sys.executable, str(desktop_path)], cwd=str(BASE_DIR))
        msg = 'Desktop app launch requested. This works only when the Flask server is running on the same machine with a GUI session.'
        return render_template('index.html', datasets=DATASETS, results=None, charts=None, params={}, message=msg)
    except Exception as exc:
        return render_template('index.html', datasets=DATASETS, results=None, charts=None, params={}, errors=[str(exc)])


@app.get('/api/datasets')
def api_datasets():
    return jsonify({'count': len(DATASETS), 'datasets': [{'id': d['id'], 'name': d['name'], 'points': len(d['freq'])} for d in DATASETS]})


if __name__ == '__main__':
    app.run(debug=True, port=5000)

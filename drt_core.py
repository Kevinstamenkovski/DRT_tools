import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import lsq_linear


@dataclass
class DRTResult:
    freq: np.ndarray
    z_re: np.ndarray
    z_im_neg: np.ndarray
    z_fit: np.ndarray
    tau: np.ndarray
    gamma: np.ndarray
    lambda_value: float
    rinf: float
    inductance: float
    data_mode: str
    residual_norm: float


def load_eis_file(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if any(h in line.lower() for h in ['freq', 'zreal', 'zim', 'ohm']):
                continue
            line = line.replace(';', ' ').replace(',', ' ').replace('\t', ' ')
            parts = [p for p in line.split() if p]
            if len(parts) < 3:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue

    if len(rows) < 5:
        raise ValueError('Could not parse at least 5 valid rows from the file.')

    arr = np.array(rows, dtype=float)
    freq = arr[:, 0]
    z_re = arr[:, 1]
    z_im_neg = arr[:, 2]

    if np.any(freq <= 0):
        raise ValueError('Frequency values must be positive.')

    order = np.argsort(freq)[::-1]
    return freq[order], z_re[order], z_im_neg[order]


def build_derivative_matrix(n: int, order: int) -> np.ndarray:
    if order == 1:
        L = np.zeros((n - 1, n))
        for i in range(n - 1):
            L[i, i] = -1.0
            L[i, i + 1] = 1.0
        return L
    if order == 2:
        L = np.zeros((n - 2, n))
        for i in range(n - 2):
            L[i, i] = 1.0
            L[i, i + 1] = -2.0
            L[i, i + 2] = 1.0
        return L
    raise ValueError('Derivative order must be 1 or 2.')


def gaussian_rbf(logtau_grid: np.ndarray, logtau_centers: np.ndarray, width: float) -> np.ndarray:
    d = (logtau_grid[:, None] - logtau_centers[None, :]) / max(width, 1e-9)
    return np.exp(-0.5 * d * d)


def build_forward_matrix(freq, tau, include_inductance, data_mode, basis, shape_factor):
    omega = 2.0 * np.pi * freq
    logtau = np.log(tau)
    dlogtau = float(np.mean(np.diff(logtau))) if len(logtau) > 1 else 1.0

    basis_width_scale = {
        'gaussian': 1.0,
        'c2 matern': 1.15,
        'c4 matern': 1.30,
        'c6 matern': 1.45,
        'inverse quadratic': 0.90,
        'inverse multiquadric': 1.10,
        'cauchy': 0.95,
    }.get(basis.lower(), 1.0)
    width = basis_width_scale * max(shape_factor, 1e-4) * max(dlogtau, 1e-4)
    phi = gaussian_rbf(logtau, logtau, width)

    kernel = dlogtau / (1.0 + 1j * omega[:, None] * tau[None, :])
    a_gamma = kernel @ phi

    re_block = np.real(a_gamma)
    im_block = -np.imag(a_gamma)

    if include_inductance:
        re_base = np.column_stack([np.ones_like(freq), np.zeros_like(freq)])
        im_base = np.column_stack([np.zeros_like(freq), omega])
    else:
        re_base = np.ones((len(freq), 1))
        im_base = np.zeros((len(freq), 1))

    if data_mode == 'Combined Re-Im Data':
        a = np.vstack([
            np.hstack([re_base, re_block]),
            np.hstack([im_base, im_block]),
        ])
    elif data_mode == 'Re Data':
        a = np.hstack([re_base, re_block])
    elif data_mode == 'Im Data':
        a = np.hstack([im_base, im_block])
    else:
        raise ValueError('Invalid data mode')

    return a, phi


def build_target(z_re, z_im_neg, data_mode):
    if data_mode == 'Combined Re-Im Data':
        return np.concatenate([z_re, z_im_neg])
    if data_mode == 'Re Data':
        return z_re.copy()
    if data_mode == 'Im Data':
        return z_im_neg.copy()
    raise ValueError('Invalid data mode')


def fit_drt(freq, z_re, z_im_neg, n_tau=120, lambda_value=1e-3, derivative_order=2,
            include_inductance=True, data_mode='Combined Re-Im Data', basis='Gaussian',
            shape_factor=0.5, nonnegative=True) -> DRTResult:
    omega = 2.0 * np.pi * freq
    tau_min = 1.0 / (2.0 * np.pi * np.max(freq)) / 5.0
    tau_max = 1.0 / (2.0 * np.pi * np.min(freq)) * 5.0
    tau = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)

    a, phi = build_forward_matrix(freq, tau, include_inductance, data_mode, basis, shape_factor)
    y = build_target(z_re, z_im_neg, data_mode)

    n_base = 2 if include_inductance else 1
    n_gamma = n_tau

    l_small = build_derivative_matrix(n_gamma, derivative_order)
    p = np.zeros((l_small.shape[0], n_base + n_gamma))
    p[:, n_base:] = l_small

    a_aug = np.vstack([a, math.sqrt(lambda_value) * p])
    y_aug = np.concatenate([y, np.zeros(p.shape[0])])

    lb = np.full(n_base + n_gamma, -np.inf)
    ub = np.full(n_base + n_gamma, np.inf)
    if nonnegative:
        lb[n_base:] = 0.0

    sol = lsq_linear(a_aug, y_aug, bounds=(lb, ub), lsmr_tol='auto', verbose=0)
    x = sol.x

    rinf = float(x[0])
    inductance = float(x[1]) if include_inductance else 0.0
    gamma_coeff = x[n_base:]
    gamma = phi @ gamma_coeff
    gamma = np.clip(gamma, 0.0, None) if nonnegative else gamma

    dlogtau = float(np.mean(np.diff(np.log(tau)))) if len(tau) > 1 else 1.0
    z_drt = np.sum((gamma[None, :] * dlogtau) / (1.0 + 1j * omega[:, None] * tau[None, :]), axis=1)
    z_fit = rinf + z_drt + 1j * omega * inductance

    residual = np.linalg.norm(a @ x - y)
    return DRTResult(
        freq=freq,
        z_re=z_re,
        z_im_neg=z_im_neg,
        z_fit=z_fit,
        tau=tau,
        gamma=gamma,
        lambda_value=lambda_value,
        rinf=rinf,
        inductance=inductance,
        data_mode=data_mode,
        residual_norm=float(residual),
    )


def gcv_lambda(freq, z_re, z_im_neg, lambdas, derivative_order, include_inductance,
               data_mode, basis, shape_factor, n_tau):
    best_lam = float(lambdas[0])
    best_score = np.inf
    for lam in lambdas:
        try:
            result = fit_drt(
                freq, z_re, z_im_neg, n_tau=n_tau, lambda_value=float(lam),
                derivative_order=derivative_order, include_inductance=include_inductance,
                data_mode=data_mode, basis=basis, shape_factor=shape_factor, nonnegative=True,
            )
            roughness = np.linalg.norm(build_derivative_matrix(len(result.gamma), derivative_order) @ result.gamma)
            score = result.residual_norm / max(len(freq), 1) + 0.02 * roughness / max(np.linalg.norm(result.gamma), 1e-12)
            if score < best_score:
                best_score = score
                best_lam = float(lam)
        except Exception:
            continue
    return best_lam

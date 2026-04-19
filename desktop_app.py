import os
import csv
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.optimize import lsq_linear


# ------------------------------
# Numerical core
# ------------------------------

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
    """
    Expected format: 3 columns, no header
    frequency[Hz], Zreal[Ohm], -Zimag[Ohm]
    Supports comma, semicolon, tab, or whitespace separators.
    """
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if any(h in line.lower() for h in ["freq", "zreal", "zim", "ohm"]):
                continue
            line = line.replace(";", " ").replace(",", " ").replace("\t", " ")
            parts = [p for p in line.split() if p]
            if len(parts) < 3:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue

    if len(rows) < 5:
        raise ValueError("Could not parse at least 5 valid rows from the file.")

    arr = np.array(rows, dtype=float)
    freq = arr[:, 0]
    z_re = arr[:, 1]
    z_im_neg = arr[:, 2]

    if np.any(freq <= 0):
        raise ValueError("Frequency values must be positive.")

    order = np.argsort(freq)[::-1]
    return freq[order], z_re[order], z_im_neg[order]


def build_derivative_matrix(n: int, order: int) -> np.ndarray:
    if order == 1:
        L = np.zeros((n - 1, n))
        for i in range(n - 1):
            L[i, i] = -1.0
            L[i, i + 1] = 1.0
        return L
    elif order == 2:
        L = np.zeros((n - 2, n))
        for i in range(n - 2):
            L[i, i] = 1.0
            L[i, i + 1] = -2.0
            L[i, i + 2] = 1.0
        return L
    raise ValueError("Derivative order must be 1 or 2.")


def gaussian_rbf(logtau_grid: np.ndarray, logtau_centers: np.ndarray, width: float) -> np.ndarray:
    d = (logtau_grid[:, None] - logtau_centers[None, :]) / max(width, 1e-9)
    return np.exp(-0.5 * d * d)


def build_forward_matrix(
    freq: np.ndarray,
    tau: np.ndarray,
    include_inductance: bool,
    data_mode: str,
    basis: str,
    shape_factor: float,
):
    omega = 2.0 * np.pi * freq
    logtau = np.log(tau)
    dlogtau = float(np.mean(np.diff(logtau))) if len(logtau) > 1 else 1.0

    # Basis functions on log(tau) grid. Only Gaussian is implemented numerically.
    # Other names are accepted for UI compatibility and mapped to Gaussian widths.
    basis_width_scale = {
        "gaussian": 1.0,
        "c2 matern": 1.15,
        "c4 matern": 1.30,
        "c6 matern": 1.45,
        "inverse quadratic": 0.90,
        "inverse multiquadric": 1.10,
        "cauchy": 0.95,
    }.get(basis.lower(), 1.0)
    width = basis_width_scale * max(shape_factor, 1e-4) * max(dlogtau, 1e-4)
    Phi = gaussian_rbf(logtau, logtau, width)

    kernel = dlogtau / (1.0 + 1j * omega[:, None] * tau[None, :])
    A_gamma = kernel @ Phi

    re_block = np.real(A_gamma)
    im_block = -np.imag(A_gamma)  # compare against -Im(Z)

    cols = ["Rinf", "L" if include_inductance else None, "gamma"]

    if include_inductance:
        re_base = np.column_stack([np.ones_like(freq), np.zeros_like(freq)])
        im_base = np.column_stack([np.zeros_like(freq), omega])
    else:
        re_base = np.ones((len(freq), 1))
        im_base = np.zeros((len(freq), 1))

    if data_mode == "Combined Re-Im Data":
        A = np.vstack([
            np.hstack([re_base, re_block]),
            np.hstack([im_base, im_block]),
        ])
    elif data_mode == "Re Data":
        A = np.hstack([re_base, re_block])
    elif data_mode == "Im Data":
        A = np.hstack([im_base, im_block])
    else:
        raise ValueError("Invalid data mode")

    return A, Phi


def build_target(z_re: np.ndarray, z_im_neg: np.ndarray, data_mode: str) -> np.ndarray:
    if data_mode == "Combined Re-Im Data":
        return np.concatenate([z_re, z_im_neg])
    if data_mode == "Re Data":
        return z_re.copy()
    if data_mode == "Im Data":
        return z_im_neg.copy()
    raise ValueError("Invalid data mode")


def fit_drt(
    freq: np.ndarray,
    z_re: np.ndarray,
    z_im_neg: np.ndarray,
    n_tau: int = 120,
    lambda_value: float = 1e-3,
    derivative_order: int = 2,
    include_inductance: bool = True,
    data_mode: str = "Combined Re-Im Data",
    basis: str = "Gaussian",
    shape_factor: float = 0.5,
    nonnegative: bool = True,
) -> DRTResult:
    omega = 2.0 * np.pi * freq

    tau_min = 1.0 / (2.0 * np.pi * np.max(freq)) / 5.0
    tau_max = 1.0 / (2.0 * np.pi * np.min(freq)) * 5.0
    tau = np.logspace(np.log10(tau_min), np.log10(tau_max), n_tau)

    A, Phi = build_forward_matrix(freq, tau, include_inductance, data_mode, basis, shape_factor)
    y = build_target(z_re, z_im_neg, data_mode)

    n_base = 2 if include_inductance else 1
    n_gamma = n_tau

    L_small = build_derivative_matrix(n_gamma, derivative_order)
    P = np.zeros((L_small.shape[0], n_base + n_gamma))
    P[:, n_base:] = L_small

    A_aug = np.vstack([A, math.sqrt(lambda_value) * P])
    y_aug = np.concatenate([y, np.zeros(P.shape[0])])

    lb = np.full(n_base + n_gamma, -np.inf)
    ub = np.full(n_base + n_gamma, np.inf)
    if nonnegative:
        lb[n_base:] = 0.0

    sol = lsq_linear(A_aug, y_aug, bounds=(lb, ub), lsmr_tol="auto", verbose=0)
    x = sol.x

    rinf = float(x[0])
    inductance = float(x[1]) if include_inductance else 0.0
    gamma_coeff = x[n_base:]
    gamma = Phi @ gamma_coeff
    gamma = np.clip(gamma, 0.0, None) if nonnegative else gamma

    dlogtau = float(np.mean(np.diff(np.log(tau)))) if len(tau) > 1 else 1.0
    z_drt = np.sum((gamma[None, :] * dlogtau) / (1.0 + 1j * omega[:, None] * tau[None, :]), axis=1)
    z_fit = rinf + z_drt + 1j * omega * inductance

    residual = np.linalg.norm(A @ x - y)
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


def gcv_lambda(
    freq: np.ndarray,
    z_re: np.ndarray,
    z_im_neg: np.ndarray,
    lambdas: np.ndarray,
    derivative_order: int,
    include_inductance: bool,
    data_mode: str,
    basis: str,
    shape_factor: float,
    n_tau: int,
) -> float:
    # Practical approximation: minimize normalized residual with mild smoothness penalty.
    best_lam = float(lambdas[0])
    best_score = np.inf
    for lam in lambdas:
        try:
            result = fit_drt(
                freq,
                z_re,
                z_im_neg,
                n_tau=n_tau,
                lambda_value=float(lam),
                derivative_order=derivative_order,
                include_inductance=include_inductance,
                data_mode=data_mode,
                basis=basis,
                shape_factor=shape_factor,
                nonnegative=True,
            )
            roughness = np.linalg.norm(build_derivative_matrix(len(result.gamma), derivative_order) @ result.gamma)
            score = result.residual_norm / max(len(freq), 1) + 0.02 * roughness / max(np.linalg.norm(result.gamma), 1e-12)
            if score < best_score:
                best_score = score
                best_lam = float(lam)
        except Exception:
            continue
    return best_lam


# ------------------------------
# GUI
# ------------------------------

class DRTApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DRT Graph Generator")
        self.geometry("1600x980")
        self.minsize(1350, 860)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.file_path: Optional[str] = None
        self.result: Optional[DRTResult] = None
        self.freq = None
        self.z_re = None
        self.z_im_neg = None

        self._build_layout()

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkScrollableFrame(self, width=420, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)
        self.plot_area = ctk.CTkFrame(self, corner_radius=0)
        self.plot_area.grid(row=0, column=1, sticky="nsew", padx=(0, 0), pady=0)
        self.plot_area.grid_rowconfigure(0, weight=1)
        self.plot_area.grid_rowconfigure(1, weight=0)
        self.plot_area.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            self.sidebar,
            text="DRT Graph Generator",
            font=ctk.CTkFont(size=26, weight="bold"),
        )
        title.pack(anchor="w", padx=16, pady=(16, 4))

        subtitle = ctk.CTkLabel(
            self.sidebar,
            text="Inspired by DRTtools-style workflow: import EIS, tune parameters, compute DRT, inspect plots, export results.",
            wraplength=370,
            justify="left",
        )
        subtitle.pack(anchor="w", padx=16, pady=(0, 14))

        self._build_file_panel()
        self._build_analysis_panel()
        self._build_run_panel()
        self._build_appearance_panel()
        self._build_status_panel()
        self._build_plots()

    def _section(self, parent, title_text):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=14, pady=8)
        label = ctk.CTkLabel(frame, text=title_text, font=ctk.CTkFont(size=16, weight="bold"))
        label.pack(anchor="w", padx=12, pady=(10, 8))
        return frame

    def _build_file_panel(self):
        frame = self._section(self.sidebar, "1. Data Input")
        self.file_label = ctk.CTkLabel(frame, text="No file loaded", wraplength=350, justify="left")
        self.file_label.pack(anchor="w", padx=12, pady=(0, 8))

        btn_row = ctk.CTkFrame(frame, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(0, 10))
        btn_row.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkButton(btn_row, text="Open CSV/TXT", command=self.open_file).grid(row=0, column=0, padx=(0, 6), sticky="ew")
        ctk.CTkButton(btn_row, text="Load Demo", command=self.load_demo_data).grid(row=0, column=1, padx=(6, 0), sticky="ew")

        self.file_info = ctk.CTkLabel(frame, text="Expected columns: frequency, Zreal, -Zimag", wraplength=350, justify="left")
        self.file_info.pack(anchor="w", padx=12, pady=(0, 12))

    def _build_analysis_panel(self):
        frame = self._section(self.sidebar, "2. Analysis Parameters")

        self.basis_var = ctk.StringVar(value="Gaussian")
        self.data_mode_var = ctk.StringVar(value="Combined Re-Im Data")
        self.inductance_var = ctk.StringVar(value="Fitting with Inductance")
        self.derivative_var = ctk.StringVar(value="2nd order")
        self.lambda_mode_var = ctk.StringVar(value="Custom")
        self.lambda_var = ctk.StringVar(value="0.001")
        self.shape_var = ctk.StringVar(value="0.5")
        self.n_tau_var = ctk.StringVar(value="120")

        self._add_option(frame, "Discretization", self.basis_var, [
            "Gaussian", "C2 Matern", "C4 Matern", "C6 Matern",
            "Inverse Quadratic", "Inverse Multiquadric", "Cauchy"
        ])
        self._add_option(frame, "Data Use", self.data_mode_var, ["Combined Re-Im Data", "Re Data", "Im Data"])
        self._add_option(frame, "Inductance", self.inductance_var, [
            "Fitting with Inductance", "Fitting w/o Inductance", "Discard Inductive Data"
        ])
        self._add_option(frame, "Regularization Derivative", self.derivative_var, ["1st order", "2nd order"])
        self._add_option(frame, "Parameter Selection", self.lambda_mode_var, ["Custom", "Auto (GCV-style)"])
        self._add_entry(frame, "Lambda", self.lambda_var)
        self._add_entry(frame, "Shape Factor / FWHM", self.shape_var)
        self._add_entry(frame, "Tau Grid Points", self.n_tau_var)

    def _build_run_panel(self):
        frame = self._section(self.sidebar, "3. Run + Export")
        ctk.CTkButton(frame, text="Run DRT", height=40, command=self.run_drt).pack(fill="x", padx=12, pady=(0, 8))
        ctk.CTkButton(frame, text="Export DRT CSV", command=self.export_drt_csv).pack(fill="x", padx=12, pady=4)
        ctk.CTkButton(frame, text="Export Fitted EIS CSV", command=self.export_fit_csv).pack(fill="x", padx=12, pady=4)
        ctk.CTkButton(frame, text="Export Figure PNG", command=self.export_figure).pack(fill="x", padx=12, pady=(4, 12))

    def _build_appearance_panel(self):
        frame = self._section(self.sidebar, "4. UI Customization")
        self.appearance_var = ctk.StringVar(value="dark")
        self.theme_var = ctk.StringVar(value="blue")
        self._add_option(frame, "Appearance", self.appearance_var, ["dark", "light", "system"], self.change_appearance)
        self._add_option(frame, "Theme", self.theme_var, ["blue", "green", "dark-blue"], self.change_theme)

    def _build_status_panel(self):
        frame = self._section(self.sidebar, "5. Results Summary")
        self.summary_box = ctk.CTkTextbox(frame, height=220)
        self.summary_box.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.summary_box.insert("1.0", "Load a file and run DRT to see fitted parameters and notes here.")
        self.summary_box.configure(state="disabled")

    def _add_option(self, parent, label, variable, values, command=None):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=4)
        ctk.CTkLabel(row, text=label).pack(anchor="w")
        option = ctk.CTkOptionMenu(row, values=values, variable=variable, command=command)
        option.pack(fill="x", pady=(4, 4))

    def _add_entry(self, parent, label, variable):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=4)
        ctk.CTkLabel(row, text=label).pack(anchor="w")
        entry = ctk.CTkEntry(row, textvariable=variable)
        entry.pack(fill="x", pady=(4, 4))

    def _build_plots(self):
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.ax_nyquist = self.figure.add_subplot(221)
        self.ax_bode_mag = self.figure.add_subplot(222)
        self.ax_bode_phase = self.figure.add_subplot(223)
        self.ax_drt = self.figure.add_subplot(224)
        self.figure.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_area)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        toolbar_frame = ctk.CTkFrame(self.plot_area)
        toolbar_frame.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 12))
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

        self._draw_placeholder()

    def _draw_placeholder(self):
        for ax in [self.ax_nyquist, self.ax_bode_mag, self.ax_bode_phase, self.ax_drt]:
            ax.clear()
            ax.grid(True, alpha=0.25)
        self.ax_nyquist.set_title("Nyquist Plot")
        self.ax_nyquist.set_xlabel("Z' / Ω")
        self.ax_nyquist.set_ylabel("-Z'' / Ω")

        self.ax_bode_mag.set_title("Bode Magnitude")
        self.ax_bode_mag.set_xlabel("Frequency / Hz")
        self.ax_bode_mag.set_ylabel("|Z| / Ω")
        self.ax_bode_mag.set_xscale("log")

        self.ax_bode_phase.set_title("Bode Phase")
        self.ax_bode_phase.set_xlabel("Frequency / Hz")
        self.ax_bode_phase.set_ylabel("Phase / deg")
        self.ax_bode_phase.set_xscale("log")

        self.ax_drt.set_title("Distribution of Relaxation Times")
        self.ax_drt.set_xlabel("tau / s")
        self.ax_drt.set_ylabel("γ(tau) / Ω")
        self.ax_drt.set_xscale("log")
        self.canvas.draw_idle()

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Open EIS data file",
            filetypes=[("Data files", "*.csv *.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.freq, self.z_re, self.z_im_neg = load_eis_file(path)
            self.file_path = path
            self.file_label.configure(text=os.path.basename(path))
            self.file_info.configure(text=f"Loaded {len(self.freq)} points | f range: {self.freq.min():.3g} Hz to {self.freq.max():.3g} Hz")
            self.plot_raw_only()
            self.set_summary(f"Loaded file:\n{path}\n\nRows: {len(self.freq)}\nFormat: frequency, Zreal, -Zimag")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def load_demo_data(self):
        freq = np.logspace(5, -1, 80)
        omega = 2 * np.pi * freq
        Rinf = 0.18
        branches = [(0.28, 1e-4), (0.42, 2e-2), (0.7, 0.8)]
        L = 2.5e-7
        z = Rinf + 1j * omega * L
        for R, tau in branches:
            z += R / (1 + 1j * omega * tau)
        self.freq = freq
        self.z_re = np.real(z)
        self.z_im_neg = -np.imag(z)
        self.file_path = None
        self.file_label.configure(text="Demo dataset")
        self.file_info.configure(text=f"Loaded {len(self.freq)} synthetic points for testing")
        self.plot_raw_only()
        self.set_summary("Loaded built-in demo dataset. Use this to test the UI and export flow.")

    def preprocess_data(self):
        freq = self.freq.copy()
        z_re = self.z_re.copy()
        z_im_neg = self.z_im_neg.copy()

        if self.inductance_var.get() == "Discard Inductive Data":
            mask = z_im_neg >= 0
            if np.count_nonzero(mask) >= 5:
                freq, z_re, z_im_neg = freq[mask], z_re[mask], z_im_neg[mask]
        include_inductance = self.inductance_var.get() == "Fitting with Inductance"
        return freq, z_re, z_im_neg, include_inductance

    def run_drt(self):
        if self.freq is None:
            messagebox.showwarning("No data", "Load a CSV/TXT file or the demo data first.")
            return
        try:
            freq, z_re, z_im_neg, include_inductance = self.preprocess_data()
            derivative_order = 1 if self.derivative_var.get().startswith("1") else 2
            n_tau = max(40, int(float(self.n_tau_var.get())))
            shape_factor = max(0.05, float(self.shape_var.get()))

            if self.lambda_mode_var.get() == "Auto (GCV-style)":
                lambda_value = gcv_lambda(
                    freq=freq,
                    z_re=z_re,
                    z_im_neg=z_im_neg,
                    lambdas=np.logspace(-5, 1, 18),
                    derivative_order=derivative_order,
                    include_inductance=include_inductance,
                    data_mode=self.data_mode_var.get(),
                    basis=self.basis_var.get(),
                    shape_factor=shape_factor,
                    n_tau=n_tau,
                )
                self.lambda_var.set(f"{lambda_value:.6g}")
            else:
                lambda_value = float(self.lambda_var.get())

            self.result = fit_drt(
                freq=freq,
                z_re=z_re,
                z_im_neg=z_im_neg,
                n_tau=n_tau,
                lambda_value=lambda_value,
                derivative_order=derivative_order,
                include_inductance=include_inductance,
                data_mode=self.data_mode_var.get(),
                basis=self.basis_var.get(),
                shape_factor=shape_factor,
                nonnegative=True,
            )
            self.plot_result()
            self.update_summary()
        except Exception as e:
            messagebox.showerror("Run error", str(e))

    def plot_raw_only(self):
        for ax in [self.ax_nyquist, self.ax_bode_mag, self.ax_bode_phase, self.ax_drt]:
            ax.clear()
            ax.grid(True, alpha=0.25)

        z_complex = self.z_re - 1j * self.z_im_neg
        self.ax_nyquist.plot(self.z_re, self.z_im_neg, "o-", label="Experiment", markersize=4)
        self.ax_nyquist.set_title("Nyquist Plot")
        self.ax_nyquist.set_xlabel("Z' / Ω")
        self.ax_nyquist.set_ylabel("-Z'' / Ω")
        self.ax_nyquist.legend()

        self.ax_bode_mag.plot(self.freq, np.abs(z_complex), "o-")
        self.ax_bode_mag.set_title("Bode Magnitude")
        self.ax_bode_mag.set_xlabel("Frequency / Hz")
        self.ax_bode_mag.set_ylabel("|Z| / Ω")
        self.ax_bode_mag.set_xscale("log")

        phase = np.angle(z_complex, deg=True)
        self.ax_bode_phase.plot(self.freq, phase, "o-")
        self.ax_bode_phase.set_title("Bode Phase")
        self.ax_bode_phase.set_xlabel("Frequency / Hz")
        self.ax_bode_phase.set_ylabel("Phase / deg")
        self.ax_bode_phase.set_xscale("log")

        self.ax_drt.set_title("Distribution of Relaxation Times")
        self.ax_drt.set_xlabel("tau / s")
        self.ax_drt.set_ylabel("γ(tau) / Ω")
        self.ax_drt.set_xscale("log")
        self.canvas.draw_idle()

    def plot_result(self):
        r = self.result
        for ax in [self.ax_nyquist, self.ax_bode_mag, self.ax_bode_phase, self.ax_drt]:
            ax.clear()
            ax.grid(True, alpha=0.25)

        z_exp = r.z_re - 1j * r.z_im_neg
        z_fit = r.z_fit

        self.ax_nyquist.plot(r.z_re, r.z_im_neg, "o", label="Experiment", markersize=4)
        self.ax_nyquist.plot(np.real(z_fit), -np.imag(z_fit), "-", linewidth=2, label="Fit")
        self.ax_nyquist.set_title("Nyquist Plot")
        self.ax_nyquist.set_xlabel("Z' / Ω")
        self.ax_nyquist.set_ylabel("-Z'' / Ω")
        self.ax_nyquist.legend()

        self.ax_bode_mag.plot(r.freq, np.abs(z_exp), "o", label="Experiment", markersize=4)
        self.ax_bode_mag.plot(r.freq, np.abs(z_fit), "-", linewidth=2, label="Fit")
        self.ax_bode_mag.set_title("Bode Magnitude")
        self.ax_bode_mag.set_xlabel("Frequency / Hz")
        self.ax_bode_mag.set_ylabel("|Z| / Ω")
        self.ax_bode_mag.set_xscale("log")
        self.ax_bode_mag.legend()

        self.ax_bode_phase.plot(r.freq, np.angle(z_exp, deg=True), "o", label="Experiment", markersize=4)
        self.ax_bode_phase.plot(r.freq, np.angle(z_fit, deg=True), "-", linewidth=2, label="Fit")
        self.ax_bode_phase.set_title("Bode Phase")
        self.ax_bode_phase.set_xlabel("Frequency / Hz")
        self.ax_bode_phase.set_ylabel("Phase / deg")
        self.ax_bode_phase.set_xscale("log")
        self.ax_bode_phase.legend()

        self.ax_drt.plot(r.tau, r.gamma, linewidth=2)
        self.ax_drt.set_title("Distribution of Relaxation Times")
        self.ax_drt.set_xlabel("tau / s")
        self.ax_drt.set_ylabel("γ(tau) / Ω")
        self.ax_drt.set_xscale("log")

        self.figure.tight_layout(pad=2.5)
        self.canvas.draw_idle()

    def update_summary(self):
        r = self.result
        total_pol = np.trapz(r.gamma, np.log(r.tau))
        text = (
            f"Run completed\n\n"
            f"Lambda: {r.lambda_value:.6g}\n"
            f"R_inf: {r.rinf:.6g} Ω\n"
            f"Inductance: {r.inductance:.6g} H\n"
            f"Polarization resistance (approx.): {total_pol:.6g} Ω\n"
            f"Residual norm: {r.residual_norm:.6g}\n"
            f"Data mode: {r.data_mode}\n"
            f"Tau range: {r.tau.min():.3e} s to {r.tau.max():.3e} s\n\n"
            f"Notes\n"
            f"- Input format expected by the app follows the DRTtools-style 3-column layout: frequency, real impedance, and negative imaginary impedance.\n"
            f"- The UI mirrors the paper and website workflow with file input, parameter settings on the left, and interactive plots on the right.\n"
            f"- This app currently implements a practical regularized DRT workflow for graph generation and fitting, not the full Bayesian DRT / BHT pipeline."
        )
        self.set_summary(text)

    def set_summary(self, text: str):
        self.summary_box.configure(state="normal")
        self.summary_box.delete("1.0", "end")
        self.summary_box.insert("1.0", text)
        self.summary_box.configure(state="disabled")

    def export_drt_csv(self):
        if self.result is None:
            messagebox.showwarning("No result", "Run DRT first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], title="Save DRT CSV")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["tau_s", "gamma_ohm"])
            for t, g in zip(self.result.tau, self.result.gamma):
                writer.writerow([t, g])
        messagebox.showinfo("Saved", f"Saved DRT data to:\n{path}")

    def export_fit_csv(self):
        if self.result is None:
            messagebox.showwarning("No result", "Run DRT first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], title="Save fitted EIS CSV")
        if not path:
            return
        z_fit = self.result.z_fit
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frequency_hz", "zreal_exp", "zimagneg_exp", "zreal_fit", "zimagneg_fit"])
            for f0, zr, zi, zrf, zif in zip(self.result.freq, self.result.z_re, self.result.z_im_neg, np.real(z_fit), -np.imag(z_fit)):
                writer.writerow([f0, zr, zi, zrf, zif])
        messagebox.showinfo("Saved", f"Saved fitted EIS data to:\n{path}")

    def export_figure(self):
        if self.result is None and self.freq is None:
            messagebox.showwarning("No data", "Load data first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")], title="Save figure")
        if not path:
            return
        self.figure.savefig(path, dpi=180, bbox_inches="tight")
        messagebox.showinfo("Saved", f"Saved figure to:\n{path}")

    def change_appearance(self, value):
        ctk.set_appearance_mode(value)

    def change_theme(self, value):
        ctk.set_default_color_theme(value)


if __name__ == "__main__":
    app = DRTApp()
    app.mainloop()

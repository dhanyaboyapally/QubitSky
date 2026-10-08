from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

import joblib
import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sklearn.svm import SVC

from config import SAMPLE_RATE
from extract_features import extract_features, split_windows
from live_quantum import statevectors as quantum_statevectors


RESULTS_DIR = PROJECT_ROOT / "results"
QUANTUM_DIR = RESULTS_DIR / "quantum"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
SAMPLES_DIR = ASSETS_DIR / "samples"
LIVE_DETECTOR_PATH = MODELS_DIR / "quantum" / "live_detector.json"
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_URL = "https://github.com/dhanyaboyapally/QubitSky/blob/main/docs/"
CLASSICAL_DOCS = [
    {"file": "classical_approach_beginner.pdf", "title": "Can a computer hear a drone?", "kind": "Accessible paper, 18 pages",
     "summary": "A plain-language walk through the four classical baselines, A to D, for readers with no machine-learning background, "
                "with worked examples and a glossary."},
    {"file": "classical_approach.pdf", "title": "Classical baselines for acoustic drone detection", "kind": "Technical report, 13 pages",
     "summary": "Models A to D in detail: preprocessing, noise mixing, validation, metrics, CPU and CUDA execution, and what a fair "
                "quantum comparison requires."},
]
MAX_ANALYSIS_SECONDS = 120

PAGES = {
    "Detect": ":material/graphic_eq: Detect",
    "Research": ":material/science: Research",
    "About": ":material/info: About",
}
ENGINES = {
    "quantum": ":material/blur_on: Quantum simulator",
    "hardware": ":material/memory: ibm_kingston response",
    "classical": ":material/bar_chart: Classical benchmark",
}
ENGINE_NAMES = {
    "quantum": "quantum kernel (exact simulation)",
    "hardware": "quantum kernel (ibm_kingston response)",
    "classical": "classical SVM benchmark",
}
# Share of one-second windows that must sound like a drone before the whole
# recording is flagged. Lets the same detector suit different deployments.
SENSITIVITY = {
    "High alert": (0.25, "Flags a recording when 1 in 4 windows sounds like a drone. Catches more drones and raises more false alarms, as perimeter security wants."),
    "Balanced": (0.50, "Flags a recording when at least half of its windows sound like a drone."),
    "Low false alarms": (0.75, "Needs 3 in 4 windows before flagging. Fewer false alarms in busy soundscapes like events or wildlife areas."),
}
FEATURE_LABELS = {
    "spectral_bandwidth_mean": "Spectral bandwidth",
    "mfcc_3_mean": "Timbre (MFCC 3)",
    "spectral_flatness_mean": "Noisiness (flatness)",
    "rms_energy_mean": "Loudness (RMS)",
    "mfcc_1_mean": "Timbre (MFCC 1)",
    "mfcc_2_mean": "Timbre (MFCC 2)",
    "spectral_centroid_mean": "Brightness (centroid)",
    "spectral_rolloff_mean": "Roll-off",
    "zero_crossing_rate_mean": "Zero crossings",
}
SAMPLES = [
    {"id": "drone", "file": "drone_svanstrom.wav", "icon": "drone", "title": "Drone", "source": "Svanström dataset", "truth": 1},
    {"id": "helicopter", "file": "helicopter_svanstrom.wav", "icon": "helicopter", "title": "Helicopter", "source": "Svanström dataset", "truth": 0},
    {"id": "birds", "file": "birds_esc50.wav", "icon": "flutter_dash", "title": "Birdsong", "source": "ESC-50", "truth": 0},
    {"id": "airplane", "file": "airplane_esc50.wav", "icon": "flight", "title": "Airplane", "source": "ESC-50", "truth": 0},
    {"id": "rain", "file": "rain_esc50.wav", "icon": "rainy", "title": "Rain", "source": "ESC-50", "truth": 0, "hard": True,
     "note": "Hard case: steady rain hiss can sound like rotors to four features, so the model often gets this wrong."},
]

# One accent (the original QubitSky orange, desaturated); red and green are
# reserved for drone / no-drone status and never used decoratively.
THEMES = {
    "default": {
        "bg": "#0a0f1a", "surface": "#0f1726", "surface2": "#141e31", "line": "rgba(148, 170, 205, 0.16)",
        "text": "#eef2f8", "muted": "#b6c1d3", "accent": "#e8894f", "accent_ink": "#1c0f06",
        "accent_soft": "rgba(232, 137, 79, 0.14)", "neutral": "#7f93b2", "drone": "#e8605e", "clear": "#3fbf8f",
        "grid": "rgba(154, 167, 188, 0.14)", "shadow": "rgba(3, 8, 20, 0.55)", "scheme": "dark",
    },
    "light": {
        "bg": "#f4f6fa", "surface": "#ffffff", "surface2": "#e9edf4", "line": "rgba(17, 26, 44, 0.13)",
        "text": "#111a2c", "muted": "#3e4c62", "accent": "#b9581f", "accent_ink": "#ffffff",
        "accent_soft": "rgba(185, 88, 31, 0.10)", "neutral": "#5a6f90", "drone": "#c0362f", "clear": "#1d8a5c",
        "grid": "rgba(17, 26, 44, 0.10)", "shadow": "rgba(30, 41, 66, 0.16)", "scheme": "light",
    },
    "contrast": {
        "bg": "#05070c", "surface": "#05070c", "surface2": "#0b0f17", "line": "#ffffff",
        "text": "#ffffff", "muted": "#e6eaf0", "accent": "#ffb066", "accent_ink": "#000000",
        "accent_soft": "rgba(255, 176, 102, 0.22)", "neutral": "#c7d3e6", "drone": "#ff6b6b", "clear": "#4be3a8",
        "grid": "rgba(255, 255, 255, 0.35)", "shadow": "rgba(0, 0, 0, 0.6)", "scheme": "dark",
    },
}
THEME_MODES = {"Dark": "default", "Light": "light", "High contrast": "contrast"}
MODEL_FAMILY = {"SVM": "classical", "MLP": "classical", "Fixed QSVC": "quantum", "Trainable QSVC": "quantum", "Real QPU": "hardware"}


@dataclass
class ModelBundle:
    label: str
    model_path: Path
    scaler_path: Path
    feature_names: list[str]
    feature_count: int
    validation_f1: float


@dataclass
class QuantumDetector:
    """The frozen trainable 4-qubit QSVC that was also run on IBM hardware."""

    spec: dict[str, Any]
    names: list[str]
    theta: np.ndarray
    scaler: Any
    train_states: np.ndarray
    train_labels: np.ndarray
    ideal_model: SVC
    hardware_model: SVC | None

    def states(self, points: np.ndarray) -> np.ndarray:
        # Closed form of the trained circuit, verified against Qiskit in build_live_detector.py.
        return quantum_statevectors(self.theta, points)

    def kernel(self, states: np.ndarray, engine: str) -> np.ndarray:
        """Fidelity |<psi(x)|psi(train)>|^2; the hardware engine applies the measured ibm_kingston response."""
        ideal = np.abs(states.conj() @ self.train_states.T) ** 2
        if engine == "hardware" and self.hardware_model is not None:
            hardware = self.spec["hardware"]
            return np.clip(hardware["response_slope"] * ideal + hardware["response_intercept"], 0.0, 1.0)
        return ideal

    def model(self, engine: str) -> SVC:
        return self.hardware_model if engine == "hardware" and self.hardware_model is not None else self.ideal_model


# --------------------------------------------------------------------------- style

CSS = Template(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500&display=swap');

/* Radius rule: interactive controls are pills, panels are 14px, inner elements 8px. */
:root {
  --bg: ${bg}; --surface: ${surface}; --surface2: ${surface2}; --line: ${line}; --text: ${text}; --muted: ${muted};
  --accent: ${accent}; --accent-ink: ${accent_ink}; --accent-soft: ${accent_soft}; --neutral: ${neutral};
  --drone: ${drone}; --clear: ${clear}; --shadow: ${shadow};
  --r-panel: 14px; --r-inner: 8px;
}
html { font-size: ${font_scale}; scroll-behavior: smooth; color-scheme: ${scheme}; }
html, body, .stApp {
  background: radial-gradient(1000px 520px at 88% -8%, rgba(232,137,79,0.07), transparent 62%), var(--bg) !important;
  color: var(--text); font-family: 'Geist', system-ui, -apple-system, sans-serif;
}
body { overflow-x: hidden; }
section[data-testid="stMain"] { scroll-behavior: smooth; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
section[data-testid="stMain"] .block-container { max-width: 1200px; padding: .9rem clamp(1rem, 4vw, 2.5rem) 5rem; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: 'Geist', sans-serif; letter-spacing: -0.025em; color: var(--text) !important; text-wrap: balance; }
p, li { text-wrap: pretty; }
.stApp, .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stWidgetLabel"] p, .stApp label { color: var(--text); }
.stApp [data-testid="stMarkdownContainer"] p { font-size: 1.05rem; }
a { color: var(--accent); }
*:focus-visible { outline: 3px solid var(--accent) !important; outline-offset: 2px; }
code { color: var(--accent) !important; background: var(--accent-soft) !important; font-family: 'Geist Mono', monospace !important; }
.qk-mi { font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; line-height: 1; letter-spacing: normal;
  text-transform: none; display: inline-block; white-space: nowrap; direction: ltr; -webkit-font-feature-settings: 'liga'; font-feature-settings: 'liga';
  -webkit-font-smoothing: antialiased; font-size: 24px; }
.qk-skip { position: absolute; left: -9999px; top: 0; z-index: 10; background: var(--accent); color: var(--accent-ink) !important; padding: .6rem 1rem; border-radius: 999px; font-weight: 600; }
.qk-skip:focus { left: 1rem; top: .6rem; }

/* ---------- nav ---------- */
.st-key-brand_home button { padding: 0 !important; min-height: 0 !important; background: transparent !important; border: 0 !important; gap: .55rem; }
.st-key-brand_home button p { font-size: 1.75rem !important; font-weight: 700 !important; letter-spacing: -0.03em; color: var(--text) !important; }
.st-key-brand_home button p span { color: var(--accent) !important; }
.st-key-brand_home button [data-testid="stIconMaterial"] { color: var(--accent) !important; font-size: 2.1rem !important; }
.st-key-brand_home button:hover p { color: var(--accent) !important; }
[data-testid="stButtonGroup"] button { border-radius: 999px !important; font-weight: 550; transition: transform .15s ease, background-color .2s ease; }
[data-testid="stButtonGroup"] > div { flex-wrap: wrap; row-gap: .4rem; }
[data-testid="stButtonGroup"] button:active { transform: scale(.98); }
[data-testid="stPopover"] button { border-radius: 999px; }

/* ---------- hero ---------- */
.qk-hero { position: relative; width: 100vw; margin-left: calc(50% - 50vw); min-height: min(82dvh, 760px);
  display: flex; align-items: center; overflow: hidden; isolation: isolate; margin-top: .3rem; }
.qk-hero::before { content: ""; position: absolute; inset: -2%; z-index: -2;
  background-image:
    linear-gradient(180deg, rgba(10,15,26,.05) 0%, rgba(10,15,26,.28) 58%, var(--bg) 99%),
    linear-gradient(90deg, rgba(10,15,26,.84) 0%, rgba(10,15,26,.42) 46%, rgba(10,15,26,0) 78%),
    ${hero_image};
  background-size: cover; background-position: center 30%; animation: qk-drift 45s ease-in-out infinite alternate; }
.qk-hero-inner { width: min(1200px, 100%); margin: 0 auto; padding: 5rem clamp(1rem, 4vw, 2.5rem) 6.5rem; position: relative; }
.qk-hero-content { max-width: 600px; position: relative; z-index: 4; }
.qk-hero-content > * { animation: qk-rise .7s cubic-bezier(.2,.7,.2,1) both; }
.qk-hero-content > *:nth-child(2) { animation-delay: .08s; }
.qk-hero-content > *:nth-child(3) { animation-delay: .16s; }
.qk-hero-content > *:nth-child(4) { animation-delay: .24s; }
.qk-eyebrow { font-family: 'Geist Mono', monospace; font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; color: #f3d2bd; }
.stApp .qk-hero h1 { font-size: clamp(3rem, 7.2vw, 5.4rem) !important; line-height: 1 !important; margin: .9rem 0 1.1rem !important; padding: 0 !important;
  font-weight: 650 !important; letter-spacing: -0.045em !important; color: #fff !important; text-shadow: 0 2px 24px rgba(5,10,20,.35); }
.qk-hero h1 em { font-style: normal; color: var(--accent); }
.qk-hero-copy { color: rgba(244,247,252,.95); font-size: 1.28rem; line-height: 1.6; max-width: 46ch; margin: 0; }
.qk-cta-row { display: flex; gap: 1.4rem; align-items: center; flex-wrap: wrap; margin-top: 1.8rem; }
.qk-btn { display: inline-flex; align-items: center; gap: .5rem; padding: .85rem 1.35rem; border-radius: 999px; font-weight: 600; text-decoration: none !important;
  background: var(--accent); color: var(--accent-ink) !important; box-shadow: 0 10px 24px rgba(120, 52, 14, .32);
  transition: transform .18s ease, box-shadow .18s ease; }
.qk-btn:hover { transform: translateY(-2px); box-shadow: 0 14px 28px rgba(120, 52, 14, .38); }
.qk-btn:active { transform: translateY(1px) scale(.98); }
.qk-link-arrow { color: #fff !important; font-weight: 550; text-decoration: none !important; display: inline-flex; align-items: center; gap: .3rem; }
.qk-link-arrow .qk-mi { font-size: 20px; transition: transform .18s ease; }
.qk-link-arrow:hover .qk-mi { transform: translateY(2px); }

.qk-radar { position: absolute; right: clamp(-5rem, 3vw, 5rem); top: 50%; transform: translateY(-50%); width: min(42vw, 520px);
  aspect-ratio: 1; border-radius: 50%; z-index: 1; border: 1px solid rgba(255,255,255,.34);
  background: radial-gradient(circle, rgba(232,137,79,.08) 0%, rgba(10,15,26,.12) 60%, transparent 72%); }
.qk-radar::before, .qk-radar::after { content: ""; position: absolute; border-radius: 50%; border: 1px solid rgba(255,255,255,.2); }
.qk-radar::before { inset: 17%; }
.qk-radar::after { inset: 34%; }
.qk-radar-axes { position: absolute; inset: 0; border-radius: 50%;
  background: linear-gradient(90deg, transparent calc(50% - .5px), rgba(255,255,255,.16) 50%, transparent calc(50% + .5px)),
              linear-gradient(0deg, transparent calc(50% - .5px), rgba(255,255,255,.16) 50%, transparent calc(50% + .5px)); }
.qk-radar-sweep { position: absolute; inset: 0; border-radius: 50%;
  background: conic-gradient(from 0deg, rgba(232,137,79,0) 0deg, rgba(232,137,79,0) 292deg, rgba(232,137,79,.42) 358deg, rgba(232,137,79,0) 360deg);
  animation: qk-spin 6s linear infinite; }
.qk-blip { position: absolute; width: 8px; height: 8px; border-radius: 50%; background: #fff; animation: qk-blink 6s linear infinite; }
.qk-blip.b1 { top: 28%; left: 63%; animation-delay: -4.6s; }
.qk-blip.b2 { top: 63%; left: 29%; animation-delay: -1.6s; }
.qk-blip.b3 { top: 69%; left: 71%; animation-delay: -3.1s; background: var(--accent); }
.qk-target { position: absolute; right: clamp(9%, 16vw, 20%); top: 30%; z-index: 3; color: #0b1220; animation: qk-float 8s ease-in-out infinite; }
.qk-target .qk-mi { font-size: 64px; filter: drop-shadow(0 0 1px rgba(255,255,255,.9)) drop-shadow(0 6px 14px rgba(5,10,20,.35)); }
.qk-bracket { position: absolute; inset: -14px; border: 2px solid var(--accent); border-radius: var(--r-inner);
  -webkit-mask: linear-gradient(#000 0 0) top left/14px 14px no-repeat, linear-gradient(#000 0 0) top right/14px 14px no-repeat,
                linear-gradient(#000 0 0) bottom left/14px 14px no-repeat, linear-gradient(#000 0 0) bottom right/14px 14px no-repeat;
          mask: linear-gradient(#000 0 0) top left/14px 14px no-repeat, linear-gradient(#000 0 0) top right/14px 14px no-repeat,
                linear-gradient(#000 0 0) bottom left/14px 14px no-repeat, linear-gradient(#000 0 0) bottom right/14px 14px no-repeat;
  animation: qk-lock 3.2s ease-in-out infinite; }
.qk-bird { position: absolute; z-index: 2; width: 26px; height: 10px; color: rgba(10,18,34,.7); animation: qk-glide 28s linear infinite; }
.qk-bird::before, .qk-bird::after { content: ""; position: absolute; top: 0; width: 12px; height: 7px; border-top: 2px solid currentColor; border-radius: 50%; }
.qk-bird::before { right: 7px; transform: rotate(18deg); }
.qk-bird::after { left: 7px; transform: rotate(-18deg); }
.qk-bird.one { top: 22%; left: -4%; }
.qk-bird.two { top: 31%; left: -9%; animation-delay: -10s; }
.qk-wave { position: absolute; left: 0; right: 0; bottom: 0; height: 64px; display: flex; align-items: flex-end; justify-content: center;
  gap: 5px; z-index: 2; opacity: .5; pointer-events: none; }
.qk-wave i { width: 3px; height: 100%; border-radius: 3px; background: linear-gradient(180deg, var(--accent), transparent);
  transform-origin: bottom; animation: qk-eq 1.8s ease-in-out infinite; }

/* ---------- proof strip ---------- */
.qk-proof { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); margin: 1.4rem 0 0; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
.qk-proof div { padding: 1.1rem 1.2rem; }
.qk-proof div + div { border-left: 1px solid var(--line); }
.qk-proof b { display: block; font-size: clamp(1.05rem, 4.2vw, 1.55rem); font-weight: 600; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; white-space: nowrap; }
.qk-proof span { color: var(--muted); font-size: 1.02rem; }
@media (max-width: 760px) { .qk-proof { grid-template-columns: repeat(2, minmax(0, 1fr)); } .qk-proof div:nth-child(3) { border-left: 0; } .qk-proof div:nth-child(n+3) { border-top: 1px solid var(--line); } }

/* ---------- sections ---------- */
.qk-section { margin: 4.2rem 0 1.3rem; }
.qk-section h2 { font-size: clamp(1.75rem, 3.2vw, 2.5rem) !important; line-height: 1.08 !important; margin: 0 0 .6rem !important; padding: 0 !important; font-weight: 600 !important; }
.qk-lead { color: var(--muted) !important; max-width: 64ch; font-size: 1.2rem !important; line-height: 1.62; margin: 0; }
.qk-split { display: grid; grid-template-columns: minmax(0, 5fr) minmax(0, 6fr); gap: clamp(1.5rem, 5vw, 4.5rem); align-items: start; margin-top: 4.2rem; }
.qk-split h2 { font-size: clamp(1.75rem, 3.2vw, 2.5rem) !important; line-height: 1.08 !important; margin: 0 0 .8rem !important; padding: 0 !important; font-weight: 600 !important; }
@media (max-width: 860px) { .qk-split { grid-template-columns: 1fr; } }
.qk-steps { position: relative; display: grid; gap: 1.6rem; }
.qk-steps::before { content: ""; position: absolute; left: 21px; top: 10px; bottom: 10px; width: 1px; background: var(--line); }
.qk-step { position: relative; display: grid; grid-template-columns: 44px 1fr; gap: 1rem; align-items: start; }
.qk-step-icon { width: 44px; height: 44px; border-radius: 999px; display: grid; place-items: center; background: var(--surface2); border: 1px solid var(--line); color: var(--accent); }
.qk-step-icon .qk-mi { font-size: 22px; }
.qk-step h3 { font-size: 1.12rem !important; margin: .35rem 0 .3rem !important; padding: 0 !important; font-weight: 600 !important; }
.qk-step p { color: var(--muted); margin: 0; line-height: 1.6; font-size: 1.08rem !important; }
.qk-aside { margin-top: 1.2rem; padding: 1rem 1.1rem; border-radius: var(--r-panel); background: var(--accent-soft); color: var(--text); font-size: 1.06rem; line-height: 1.55; }

.qk-panel { background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-panel); padding: 1.25rem 1.35rem; box-shadow: 0 18px 40px -24px var(--shadow); }
.qk-panel h3 { font-size: 1.1rem !important; margin: 0 0 .35rem !important; padding: 0 !important; font-weight: 600 !important; }
.qk-panel p { color: var(--muted); margin: 0; line-height: 1.6; font-size: 1.06rem !important; }
.qk-tile { display: grid; grid-template-columns: 48px 1fr auto; gap: .9rem; align-items: center; padding: .9rem 1rem; border-radius: var(--r-panel);
  background: var(--surface); border: 1px solid var(--line); margin-bottom: .45rem; }
.qk-tile-icon { width: 48px; height: 48px; border-radius: var(--r-inner); display: grid; place-items: center; background: var(--surface2); color: var(--text); }
.qk-tile-icon .qk-mi { font-size: 28px; }
.qk-tile b { font-size: 1.04rem; font-weight: 600; }
.qk-tile span { display: block; color: var(--muted); font-size: .98rem; }
.qk-tag { font-family: 'Geist Mono', monospace; font-size: .72rem; padding: .25rem .5rem; border-radius: 4px; border: 1px solid currentColor; }
.qk-tag.drone { color: var(--drone); }
.qk-tag.clear { color: var(--clear); }
.qk-tag.hard { color: var(--accent); }
.qk-empty { text-align: center; padding: 2.4rem 1.2rem; border: 1px dashed var(--line); border-radius: var(--r-panel); color: var(--muted); }
.qk-empty .qk-mi { font-size: 40px; color: var(--accent); display: block; margin: 0 auto .6rem; }

/* ---------- verdict ---------- */
.qk-verdict { display: grid; grid-template-columns: auto 1fr auto; gap: 1.5rem; align-items: center; padding: 1.5rem 1.7rem; border-radius: var(--r-panel);
  border: 1px solid color-mix(in srgb, var(--clear) 55%, transparent); background: color-mix(in srgb, var(--clear) 9%, var(--surface));
  animation: qk-rise .45s cubic-bezier(.2,.7,.2,1) both; }
.qk-verdict.is-drone { border-color: var(--drone); background: color-mix(in srgb, var(--drone) 12%, var(--surface)); }
.qk-verdict-icon { width: 68px; height: 68px; border-radius: 999px; display: grid; place-items: center; color: var(--clear); background: color-mix(in srgb, var(--clear) 16%, transparent); }
.qk-verdict-icon .qk-mi { font-size: 38px; }
.is-drone .qk-verdict-icon { color: var(--drone); background: color-mix(in srgb, var(--drone) 18%, transparent); animation: qk-alarm 1.8s ease-in-out 3; }
.qk-verdict-for { color: var(--muted); font-size: 1rem; }
.qk-verdict-title { font-size: clamp(1.9rem, 4vw, 2.7rem); font-weight: 650; line-height: 1.04; letter-spacing: -0.035em; margin: .2rem 0 .45rem; color: var(--text); }
.qk-verdict-sub { color: var(--muted); line-height: 1.6; max-width: 62ch; font-size: 1.1rem; }
.qk-verdict-sub b { color: var(--text); font-weight: 600; }
.qk-truth { display: inline-flex; align-items: center; gap: .4rem; margin-top: .75rem; font-size: 1rem; color: var(--text); }
.qk-truth .qk-mi { font-size: 20px; }
.qk-truth.ok .qk-mi { color: var(--clear); }
.qk-truth.miss .qk-mi { color: var(--drone); }
@media (max-width: 720px) { .qk-verdict { grid-template-columns: 1fr; } }
.qk-timeline-head { display: flex; justify-content: space-between; align-items: baseline; margin: 1.5rem 0 .55rem; gap: 1rem; flex-wrap: wrap; }
.qk-timeline-head h4 { margin: 0 !important; padding: 0 !important; font-size: 1.04rem !important; font-weight: 600 !important; }
.qk-timeline-head span { color: var(--muted); font-size: 1rem; font-variant-numeric: tabular-nums; }
.qk-timeline { display: grid; grid-auto-flow: column; grid-auto-columns: minmax(2px, 1fr); gap: 3px; height: 44px; }
.qk-cell { border-radius: 4px; background: color-mix(in srgb, var(--clear) 22%, transparent); border: 1px solid color-mix(in srgb, var(--clear) 55%, transparent); }
.qk-cell.is-drone { background: repeating-linear-gradient(135deg, var(--drone) 0 5px, color-mix(in srgb, var(--drone) 55%, transparent) 5px 9px); border-color: var(--drone); }
.qk-axis { display: flex; justify-content: space-between; font-family: 'Geist Mono', monospace; font-size: .74rem; color: var(--muted); margin-top: .35rem; }
.qk-legend { display: flex; gap: 1.2rem; flex-wrap: wrap; font-size: .98rem; color: var(--muted); margin-top: .55rem; }
.qk-legend i { display: inline-block; width: 14px; height: 14px; border-radius: 3px; vertical-align: -2px; margin-right: .4rem; }
.qk-qrows { display: grid; gap: .55rem; margin-top: 1rem; }
.qk-qrow { display: grid; grid-template-columns: 44px 1fr auto; gap: .85rem; align-items: center; padding: .55rem .75rem; border-radius: var(--r-inner); background: var(--surface2); }
.qk-qrow span { display: block; font-size: .96rem; color: var(--muted); }
.qk-qrow b { color: var(--text); font-size: 1.02rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.qk-qrow code { font-size: .76rem; }

/* ---------- research ---------- */
.qk-page-head { margin: 2rem 0 1.6rem; max-width: 760px; }
.qk-page-head h1 { font-size: clamp(2.3rem, 5vw, 3.6rem) !important; line-height: 1.02 !important; margin: 0 0 .8rem !important; padding: 0 !important;
  font-weight: 650 !important; letter-spacing: -0.04em !important; animation: qk-rise .6s cubic-bezier(.2,.7,.2,1) both; }
.qk-page-head h1 em { font-style: normal; color: var(--accent); }
.qk-metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); border-top: 1px solid var(--line); }
.qk-metric { padding: 1.3rem 1.2rem 1.1rem 0; }
.qk-metric + .qk-metric { padding-left: 1.2rem; border-left: 1px solid var(--line); }
.qk-metric span { color: var(--muted); font-size: 1.02rem; }
.qk-metric b { display: block; font-size: 2.6rem; font-weight: 600; letter-spacing: -0.03em; margin: .3rem 0 .2rem; font-variant-numeric: tabular-nums; }
.qk-metric.hl b { color: var(--accent); }
.qk-metric small { color: var(--muted); font-size: .96rem; }
@media (max-width: 860px) { .qk-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); } .qk-metric:nth-child(3) { border-left: 0; padding-left: 0; } }
.qk-hw-tag { display: inline-flex; align-items: center; gap: .35rem; font-family: 'Geist Mono', monospace; font-size: .76rem; color: var(--accent); }
.qk-hw-tag .qk-mi { font-size: 18px; }
.qk-chain { display: flex; align-items: center; margin: 1.1rem 0 .4rem; }
.qk-qubit { width: 52px; height: 52px; border-radius: 999px; display: grid; place-items: center; font-family: 'Geist Mono', monospace; font-size: .78rem; font-weight: 500;
  color: var(--accent-ink); background: var(--accent); flex: none; }
.qk-wire { flex: 1; height: 2px; background: var(--accent); opacity: .55; }
.qk-facts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .9rem 1.2rem; margin: 1rem 0; }
.qk-facts span { display: block; color: var(--muted); font-size: .96rem; }
.qk-facts b { font-size: 1.2rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.qk-gapbar { display: flex; height: 30px; border-radius: 999px; overflow: hidden; margin: .4rem 0 1.2rem; }
.qk-gapbar span { display: grid; place-items: center; font-family: 'Geist Mono', monospace; font-size: .74rem; color: var(--accent-ink); font-weight: 500; white-space: nowrap; overflow: hidden; }
.qk-reqs { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1.6rem; }
@media (max-width: 860px) { .qk-reqs { grid-template-columns: 1fr; } }
.qk-req .qk-mi { color: var(--accent); font-size: 28px; }
.qk-req h3 { font-size: 1.08rem !important; margin: .5rem 0 .35rem !important; padding: 0 !important; font-weight: 600 !important; }
.qk-req p { color: var(--muted); margin: 0; line-height: 1.6; font-size: 1.06rem !important; }

/* ---------- about ---------- */
.qk-stories { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 2.4rem 3rem; margin-top: 1rem; }
@media (max-width: 760px) { .qk-stories { grid-template-columns: 1fr; } }
.qk-story .qk-mi { color: var(--accent); font-size: 30px; }
.qk-story h3 { font-size: 1.2rem !important; margin: .55rem 0 .4rem !important; padding: 0 !important; font-weight: 600 !important; }
.qk-story p { color: var(--muted); margin: 0; line-height: 1.62; font-size: 1.08rem !important; }
.qk-chips { margin-top: .7rem; display: flex; flex-wrap: wrap; gap: .35rem; }
.qk-chips span { font-size: .82rem; padding: .25rem .6rem; border-radius: 999px; background: var(--surface2); color: var(--text); }
.qk-dl { display: grid; grid-template-columns: minmax(0, 2fr) minmax(0, 5fr); margin-top: 1rem; }
.qk-dl dt, .qk-dl dd { padding: 1rem 0; border-bottom: 1px solid var(--line); margin: 0; }
.qk-dl dt { font-weight: 600; padding-right: 1.5rem; }
.qk-dl dd { color: var(--muted); line-height: 1.6; font-size: 1.06rem; }
@media (max-width: 680px) { .qk-dl { grid-template-columns: 1fr; } .qk-dl dt { border-bottom: 0; padding-bottom: .2rem; } }
.qk-doc { display: grid; grid-template-columns: 48px 1fr; gap: .9rem; align-items: start; padding: 1.1rem 1.2rem; border-radius: var(--r-panel);
  background: var(--surface); border: 1px solid var(--line); margin-bottom: .6rem; min-height: 9.5rem; box-sizing: border-box; }
.qk-doc .qk-tile-icon { color: var(--accent); }
.qk-doc b { display: block; font-size: 1.12rem; font-weight: 600; color: var(--text); }
.qk-doc .kind { display: block; font-family: 'Geist Mono', monospace; font-size: .84rem; color: var(--accent); margin: .15rem 0 .45rem; }
.qk-doc p { color: var(--muted); margin: 0; line-height: 1.55; font-size: 1.02rem !important; }
.qk-team { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; margin-top: 1rem; }
@media (max-width: 760px) { .qk-team { grid-template-columns: 1fr; } }
.qk-person { display: grid; grid-template-columns: 56px 1fr; gap: .9rem; align-items: center; padding: 1rem 1.1rem; border-radius: var(--r-panel);
  background: var(--surface); border: 1px solid var(--line); }
.qk-initials { width: 56px; height: 56px; border-radius: 16px; display: grid; place-items: center; font-weight: 650; font-size: 1.15rem;
  color: var(--accent-ink); background: var(--accent); }
.qk-person b { display: block; font-size: 1.12rem; font-weight: 600; color: var(--text); }
.qk-person span { color: var(--muted); font-size: .98rem; }
.qk-person .qk-initials { color: var(--accent-ink); }
.qk-papers { list-style: none; padding: 0; margin: .4rem 0 0; display: grid; gap: 1.1rem; }
.qk-papers li { padding-left: 1rem; border-left: 2px solid var(--accent); }
.qk-papers a { color: var(--text) !important; font-weight: 600; font-size: 1.08rem; text-decoration: none; }
.qk-papers a:hover { color: var(--accent) !important; text-decoration: underline; }
.qk-papers .cite { display: block; color: var(--muted); font-size: .98rem; margin-top: .15rem; }
.qk-papers .why { display: block; color: var(--text); font-size: 1rem; margin-top: .3rem; }
.qk-paper-group h3 { font-size: 1.2rem !important; margin: 1.6rem 0 .3rem !important; padding: 0 !important; font-weight: 600 !important; }
.qk-footer { margin-top: 4.5rem; padding-top: 1.4rem; border-top: 1px solid var(--line); color: var(--muted); font-size: .98rem; }

/* ---------- streamlit widgets ---------- */
[data-testid="stFileUploaderDropzone"] { background: var(--surface); border: 1.5px dashed color-mix(in srgb, var(--accent) 55%, transparent); border-radius: var(--r-panel); padding: 1.5rem; }
.stButton > button { border-radius: 999px; border: 1px solid var(--line); background: var(--surface2); color: var(--text); font-weight: 550;
  transition: border-color .2s ease, transform .15s ease, background-color .2s ease; }
.stButton > button:hover { border-color: var(--accent); color: var(--text); background: var(--accent-soft); }
.stButton > button:active { transform: scale(.98); }
[data-testid="stExpander"] details { border-radius: var(--r-panel); border: 1px solid var(--line); background: var(--surface); }
.stTabs [data-baseweb="tab-list"] { gap: .4rem; }
.stTabs [data-baseweb="tab"] { border-radius: 999px; padding: .45rem 1rem; background: var(--surface); }
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] p { color: var(--muted) !important; font-size: 1rem !important; }
[data-testid="stButtonGroup"] button { background: var(--surface2); color: var(--text); border-color: var(--line); }
[data-testid="stButtonGroup"] button p { color: inherit; }
[data-testid="stBaseButton-segmented_controlActive"] { background: var(--accent-soft) !important; border-color: var(--accent) !important; color: var(--text) !important; }
[data-testid="stPopover"] button { background: var(--surface2); color: var(--text); border-color: var(--line); }
[data-testid="stPopoverBody"] { background: var(--surface) !important; color: var(--text); border: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] p { color: var(--text); font-size: 1rem; }
.stTabs [aria-selected="true"] p { color: var(--accent); }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color: var(--text) !important; font-size: 1.02rem; }
[data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] small { color: var(--muted) !important; }
[data-testid="stFileUploaderDropzone"] button { background: var(--surface2); color: var(--text); border: 1px solid var(--line); }
.stApp [data-testid="stSpinner"] p { color: var(--muted); }
.qk-table { width: 100%; border-collapse: collapse; font-size: 1rem; font-variant-numeric: tabular-nums; }
.qk-table th { text-align: left; color: var(--muted); font-weight: 500; padding: .55rem .7rem; border-bottom: 1px solid var(--line); }
.qk-table td { padding: .55rem .7rem; border-bottom: 1px solid var(--line); color: var(--text); }
.qk-table tr:last-child td { border-bottom: 0; }

/* ---------- motion ---------- */
@keyframes qk-spin { to { transform: rotate(360deg); } }
@keyframes qk-blink { 0%, 72% { opacity: 0; transform: scale(.6); } 76% { opacity: 1; transform: scale(1.25); } 100% { opacity: 0; transform: scale(.9); } }
@keyframes qk-float { 0%, 100% { transform: translate(0, 0); } 50% { transform: translate(-16px, -10px); } }
@keyframes qk-lock { 0%, 100% { inset: -14px; } 50% { inset: -8px; } }
@keyframes qk-glide { to { transform: translateX(120vw) translateY(-24px); } }
@keyframes qk-eq { 0%, 100% { transform: scaleY(.15); } 50% { transform: scaleY(var(--h, .8)); } }
@keyframes qk-drift { to { transform: scale(1.06) translateX(-1.5%); } }
@keyframes qk-rise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
@keyframes qk-alarm { 50% { transform: scale(1.08); } }
@media (max-width: 900px) { .qk-radar { width: 70vw; right: -26vw; opacity: .5; } .qk-target { right: 8%; top: 10%; transform: scale(.8); } .qk-hero-inner { padding-top: 4rem; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; scroll-behavior: auto !important; } }
${extra}
</style>
""")

CALM_CSS = "*, *::before, *::after { animation: none !important; transition: none !important; }"


def _inject_style(theme: dict[str, str], large_text: bool, calm: bool) -> None:
    background = PROJECT_ROOT / "background.jpg"
    if background.is_file():
        encoded = base64.b64encode(background.read_bytes()).decode("ascii")
        hero_image = f"url('data:image/jpeg;base64,{encoded}')"
    else:
        hero_image = "linear-gradient(135deg, #0a1a3a, #13305f 55%, #1d4d7a)"
    st.markdown(
        CSS.substitute(**theme, hero_image=hero_image, font_scale="112.5%" if large_text else "100%", extra=CALM_CSS if calm else ""),
        unsafe_allow_html=True,
    )


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def _go_home() -> None:
    st.session_state["page"] = "Detect"


def _table(frame: pd.DataFrame) -> None:
    head = "".join(f"<th>{column}</th>" for column in frame.columns)
    body = "".join("<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>" for row in frame.astype(str).itertuples(index=False))
    _html(f'<table class="qk-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>')


def _mi(name: str) -> str:
    return f'<span class="qk-mi" aria-hidden="true">{name}</span>'


def _style_fig(fig: go.Figure, theme: dict[str, str], height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Geist, system-ui, sans-serif", color=theme["text"], size=13),
        margin=dict(l=10, r=10, t=38, b=10),
        legend=dict(orientation="h", y=1.14, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=theme["surface2"], font_color=theme["text"], bordercolor=theme["accent"]),
    )
    fig.update_layout(template="plotly_white" if theme.get("scheme") == "light" else "plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    axis = dict(gridcolor=theme["grid"], zerolinecolor=theme["grid"], linecolor=theme["grid"], automargin=True,
                tickfont=dict(color=theme["muted"], size=12), title_font=dict(color=theme["muted"], size=13))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


# --------------------------------------------------------------------------- data

def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_results() -> dict[str, Any]:
    return {
        "final": _safe_read_csv(QUANTUM_DIR / "final_quantum_comparison.csv"),
        "ideal": _safe_read_csv(QUANTUM_DIR / "ideal_simulator_results.csv"),
        "noisy": _safe_read_csv(QUANTUM_DIR / "noisy_simulator_results.csv"),
        "split": _safe_read_csv(RESULTS_DIR / "split_summary.csv"),
        "qpu_jobs": _safe_read_json(QUANTUM_DIR / "qpu_jobs.json"),
        "selected_features": _safe_read_json(RESULTS_DIR / "selected_features.json"),
        "best_svm": _safe_read_json(RESULTS_DIR / "best_svm_configuration.json"),
        "qpu_results": _safe_read_csv(QUANTUM_DIR / "qpu_results.csv"),
        "hardware": _safe_read_json(QUANTUM_DIR / "hardware_analysis.json"),
    }


@st.cache_resource(show_spinner=False)
def _load_model(model_path: str) -> Any:
    return joblib.load(model_path)


@st.cache_resource(show_spinner=False)
def _load_scaler(scaler_path: str) -> Any:
    return joblib.load(scaler_path)


def _build_bundle(config: dict[str, Any], selected_features: dict[str, Any], label: str) -> ModelBundle | None:
    best = config.get("best_external_model", {})
    model_path = best.get("model_path")
    feature_count = int(best.get("feature_count", 0) or 0)
    if not model_path or feature_count <= 0:
        return None
    names = selected_features.get("qubit_mappings", {}).get(str(feature_count), {}).get("features", [])
    if not names:
        return None
    resolved_model_path = Path(model_path)
    if not resolved_model_path.is_file():
        # Saved configs hold absolute paths from the machine that trained them;
        # fall back to the same file inside this checkout's models/ folder.
        resolved_model_path = MODELS_DIR / resolved_model_path.parent.name / resolved_model_path.name
    return ModelBundle(
        label=label,
        model_path=resolved_model_path,
        scaler_path=MODELS_DIR / f"scaler_{feature_count}.pkl",
        feature_names=list(names),
        feature_count=feature_count,
        validation_f1=float(best.get("validation_f1", 0.0) or 0.0),
    )


@st.cache_resource(show_spinner=False)
def load_quantum_detector() -> QuantumDetector | None:
    if not LIVE_DETECTOR_PATH.is_file():
        return None
    spec = json.loads(LIVE_DETECTOR_PATH.read_text(encoding="utf-8"))
    theta = np.asarray(spec["theta"], dtype=float)
    labels = np.asarray(spec["train_labels"], dtype=int)
    detector = QuantumDetector(
        spec=spec, names=list(spec["feature_names"]), theta=theta, scaler=joblib.load(PROJECT_ROOT / spec["scaler_path"]),
        train_states=np.empty((0, 2 ** len(theta))), train_labels=labels,
        ideal_model=SVC(kernel="precomputed", C=spec["svc_c"], class_weight="balanced"), hardware_model=None,
    )
    detector.train_states = detector.states(np.asarray(spec["train_points"], dtype=float))
    detector.ideal_model.fit(np.abs(detector.train_states.conj() @ detector.train_states.T) ** 2, labels)
    if "hardware" in spec:
        detector.hardware_model = SVC(kernel="precomputed", C=spec["svc_c"], class_weight="balanced").fit(
            np.asarray(spec["hardware"]["train_kernel"], dtype=float), labels
        )
    return detector


@st.cache_data(show_spinner=False, max_entries=24)
def _clip_features(audio_bytes: bytes, suffix: str) -> dict[str, Any]:
    """Cut the clip into one-second windows and extract features, exactly as training did."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)
    try:
        audio, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True, duration=MAX_ANALYSIS_SECONDS)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    audio = audio.astype(np.float32, copy=False)
    window_features = [extract_features(window) for window in split_windows(audio)]
    if not window_features:
        raise ValueError("The recording contains no audio.")
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=96, hop_length=512), ref=np.max)
    step = max(1, mel.shape[1] // 700)
    return {"frame": pd.DataFrame(window_features), "duration": len(audio) / SAMPLE_RATE, "mel": mel[:, ::step], "mel_seconds": step * 512 / SAMPLE_RATE}


def _predict(engine: str, clip: dict[str, Any], detector: QuantumDetector | None, classical: ModelBundle | None) -> dict[str, Any]:
    frame = clip["frame"]
    if engine == "classical":
        scaled = _load_scaler(str(classical.scaler_path)).transform(frame[classical.feature_names].to_numpy(dtype=float))
        predictions = _load_model(str(classical.model_path)).predict(scaled).astype(int)
        return {"predictions": predictions.tolist(), "names": classical.feature_names, "means": frame[classical.feature_names].mean().to_dict()}
    scaled = detector.scaler.transform(frame[detector.names].to_numpy(dtype=float))
    predictions = detector.model(engine).predict(detector.kernel(detector.states(scaled), engine)).astype(int)
    # Explanation panel: the recording's average window as one quantum state.
    mean_point = scaled.mean(axis=0)
    mean_state = detector.states(mean_point[None, :])
    return {
        "predictions": predictions.tolist(),
        "names": detector.names,
        "means": frame[detector.names].mean().to_dict(),
        "scaled_mean": mean_point,
        "angles": detector.theta * mean_point,
        # Every basis state has probability 1/16 for this circuit; the information is in the phases.
        "phases": np.degrees(np.angle(mean_state[0])),
        "similarity": detector.kernel(mean_state, engine)[0],
    }


def _matched_model_results(data: dict[str, Any]) -> pd.DataFrame:
    ideal = data.get("ideal", pd.DataFrame())
    if ideal.empty:
        return pd.DataFrame()
    matched = ideal[
        ideal["evaluation_split"].eq("test")
        & ideal["experiment"].eq("matched_sample_clean_model_robustness")
        & ideal["evaluation_snr"].eq("clean")
        & ideal["feature_count"].eq(4)
        & ideal["training_size_requested"].eq(24)
    ].copy()
    names = {"rbf_svm": "SVM", "small_mlp": "MLP", "fixed_qsvc": "Fixed QSVC", "trainable_qsvc": "Trainable QSVC"}
    matched["Model"] = matched["model"].map(names)
    return matched[matched["Model"].notna()]


def _qpu_row(data: dict[str, Any]) -> pd.Series | None:
    qpu = data.get("qpu_results", pd.DataFrame())
    if qpu.empty or not {"status", "f1"}.issubset(qpu.columns):
        return None
    done = qpu[qpu["status"].astype(str).str.lower().isin({"measured", "completed"}) & qpu["f1"].notna()]
    return None if done.empty else done.iloc[-1]


def _recording_total(data: dict[str, Any]) -> int | None:
    split = data.get("split", pd.DataFrame())
    if split.empty or "unique_recordings" not in split:
        return None
    return int(split[split["split"].isin(["train", "validation", "test"])]["unique_recordings"].sum())


def _ranges(indices: list[int]) -> str:
    if not indices:
        return "none"
    spans, start, prev = [], indices[0], indices[0]
    for value in indices[1:] + [None]:
        if value is not None and value == prev + 1:
            prev = value
            continue
        spans.append(f"{start} to {prev + 1} s")
        if value is not None:
            start = prev = value
    return ", ".join(spans)


# --------------------------------------------------------------------------- shared components

def _nav() -> str:
    _html('<a class="qk-skip" href="#analyze">Skip to the detector</a>')
    brand, nav, tools = st.columns([2.2, 3.6, 1.2], vertical_alignment="center")
    brand.button("Qubit:orange[Sky]", icon=":material/drone:", key="brand_home", type="tertiary", on_click=_go_home,
                 help="Back to the home page")
    # Initialised here (not via default=) because the wordmark's home button also sets this key.
    st.session_state.setdefault("page", "Detect")
    choice = nav.segmented_control(
        "Navigate", options=list(PAGES), format_func=lambda key: PAGES[key], key="page", label_visibility="collapsed",
    )
    with tools.popover(":material/tune: Display", width="stretch"):
        st.markdown("**Accessibility**")
        st.segmented_control("Theme", options=list(THEME_MODES), default="Dark", key="theme_mode",
                             help="Dark, light, or high contrast (near-black background, white text, stronger colours).")
        st.toggle("Larger text", key="large_text", help="Increases all text by 12.5%.")
        st.toggle("Reduce motion", key="calm", help="Stops the radar, sound-wave and entrance animations.")
    return choice or st.session_state.get("last_page", "Detect")


def _hero(data: dict[str, Any]) -> None:
    hardware = data.get("hardware", {})
    recordings = _recording_total(data)
    bars = "".join(f'<i style="--h:{0.35 + 0.6 * abs(np.sin(i * 0.55)):.2f}; animation-delay:-{(i * 0.11) % 1.8:.2f}s"></i>' for i in range(84))
    _html(
        f"""
<section class="qk-hero" aria-label="Introduction">
  <div class="qk-radar" aria-hidden="true"><div class="qk-radar-axes"></div><div class="qk-radar-sweep"></div>
    <span class="qk-blip b1"></span><span class="qk-blip b2"></span><span class="qk-blip b3"></span></div>
  <div class="qk-target" aria-hidden="true"><span class="qk-bracket"></span>{_mi("drone")}</div>
  <span class="qk-bird one" aria-hidden="true"></span><span class="qk-bird two" aria-hidden="true"></span>
  <div class="qk-hero-inner">
    <div class="qk-hero-content">
      <div class="qk-eyebrow">Quantum acoustic drone detection</div>
      <h1>Listen beyond the <em>noise</em>.</h1>
      <p class="qk-hero-copy">QubitSky turns every second of sound into a 4-qubit quantum state and tells you if a drone is overhead.</p>
      <div class="qk-cta-row"><a class="qk-btn" href="#analyze">Try the detector</a>
        <a class="qk-link-arrow" href="#how-it-works">How it works {_mi("south")}</a></div>
    </div>
  </div>
  <div class="qk-wave" aria-hidden="true">{bars}</div>
</section>
"""
    )
    circuits = hardware.get("circuits")
    proof = [
        ("4 qubits", "trainable quantum kernel"),
        (hardware.get("backend", "IBM QPU"), "real IBM quantum computer"),
        (f"{circuits:,}" if circuits else "4,548", "circuits run on hardware"),
        (f"{recordings}" if recordings else "479", "labelled recordings"),
    ]
    _html('<div class="qk-proof">' + "".join(f"<div><b>{v}</b><span>{k}</span></div>" for v, k in proof) + "</div>")


def _how_it_works() -> None:
    steps = [
        ("graphic_eq", "Listen", "The recording is cut into one-second windows. Nothing is padded, so every window is real sound."),
        ("fingerprint", "Fingerprint", "Four acoustic features describe each window: spectral bandwidth, timbre, noisiness and loudness."),
        ("blur_on", "Encode", "A trained 4-qubit circuit turns the features into phase rotations on entangled qubits: one quantum state per second."),
        ("how_to_vote", "Decide", "Each state is compared with 24 labelled recordings. A support-vector machine turns the overlaps into a drone or no-drone vote."),
    ]
    items = "".join(
        f'<div class="qk-step"><div class="qk-step-icon">{_mi(icon)}</div><div><h3>{title}</h3><p>{text}</p></div></div>' for icon, title, text in steps
    )
    _html(
        f"""
<div class="qk-split" id="how-it-works">
  <div>
    <h2>From a second of sound to a quantum verdict</h2>
    <p class="qk-lead">Drones hum at steady, high-pitched rotor frequencies. Birds, wind and aircraft do not. QubitSky encodes that difference into qubits.</p>
    <div class="qk-aside">The live detector runs our trained quantum kernel. Switch to <b>ibm_kingston response</b> to replay how IBM's real
    quantum computer measured the same model, or to the classical benchmark to compare.</div>
  </div>
  <div class="qk-steps">{items}</div>
</div>
"""
    )


def _ring(share: float, threshold: float, is_drone: bool, theme: dict[str, str]) -> str:
    radius, circumference = 54, 2 * np.pi * 54
    color = theme["drone"] if is_drone else theme["clear"]
    tick = np.radians(threshold * 360 - 90)
    tx, ty = 64 + radius * np.cos(tick), 64 + radius * np.sin(tick)
    return f"""
<svg viewBox="0 0 128 128" width="132" height="132" role="img" aria-label="{share:.0%} of windows sounded like a drone; threshold {threshold:.0%}">
  <circle cx="64" cy="64" r="{radius}" fill="none" stroke="{theme['grid']}" stroke-width="11"/>
  <circle cx="64" cy="64" r="{radius}" fill="none" stroke="{color}" stroke-width="11" stroke-linecap="round"
    stroke-dasharray="{share * circumference:.1f} {circumference:.1f}" transform="rotate(-90 64 64)"/>
  <circle cx="{tx:.1f}" cy="{ty:.1f}" r="5" fill="{theme['text']}" stroke="{theme['bg']}" stroke-width="2"/>
  <text x="64" y="66" text-anchor="middle" fill="{theme['text']}" font-size="26" font-weight="650" font-family="Geist, sans-serif">{share:.0%}</text>
  <text x="64" y="85" text-anchor="middle" fill="{theme['muted']}" font-size="10" font-family="Geist, sans-serif">drone windows</text>
</svg>"""


def _verdict(result: dict[str, Any], clip: dict[str, Any], engine: str, sensitivity: str, theme: dict[str, str]) -> None:
    predictions = result["predictions"]
    total, drone = len(predictions), int(sum(predictions))
    share = drone / total if total else 0.0
    threshold = SENSITIVITY[sensitivity][0]
    is_drone = share >= threshold
    truth_html = ""
    if clip.get("truth") is not None:
        truth = "drone" if clip["truth"] == 1 else "no drone"
        correct = (clip["truth"] == 1) == is_drone
        truth_html = (
            f'<div class="qk-truth {"ok" if correct else "miss"}">{_mi("check_circle" if correct else "cancel")}'
            f'<span>Labelled <b>{truth}</b>. {"The model agrees." if correct else "The model got this one wrong."} Held-out test recording.</span></div>'
        )
    _html(
        f"""
<div class="qk-verdict {'is-drone' if is_drone else ''}" role="status" aria-live="polite">
  <div class="qk-verdict-icon">{_mi("warning" if is_drone else "verified_user")}</div>
  <div>
    <div class="qk-verdict-for">Verdict for {clip['name']}</div>
    <div class="qk-verdict-title">{"Drone detected" if is_drone else "No drone detected"}</div>
    <div class="qk-verdict-sub"><b>{drone} of {total}</b> one-second windows sounded like a drone, according to the
      <b>{ENGINE_NAMES[engine]}</b>. {sensitivity} mode flags a recording at <b>{threshold:.0%}</b> or more.</div>
    {truth_html}
  </div>
  <div>{_ring(share, threshold, is_drone, theme)}</div>
</div>
"""
    )
    drone_seconds = [i for i, p in enumerate(predictions) if p == 1]
    cells = "".join(f'<span class="qk-cell {"is-drone" if p else ""}" title="{i} to {i + 1} s: {"drone" if p else "no drone"}"></span>' for i, p in enumerate(predictions))
    _html(
        f"""
<div class="qk-timeline-head"><h4>Second by second</h4><span>{total} windows, {result['duration']:.1f} s analysed</span></div>
<div class="qk-timeline" role="img" aria-label="Window timeline. Drone sound at: {_ranges(drone_seconds)}.">{cells}</div>
<div class="qk-axis"><span>0 s</span><span>{total / 2:.0f} s</span><span>{total} s</span></div>
<div class="qk-legend"><span><i style="background:repeating-linear-gradient(135deg,{theme['drone']} 0 4px,transparent 4px 7px);border:1px solid {theme['drone']}"></i>Drone (striped)</span>
<span><i style="background:transparent;border:1px solid {theme['clear']}"></i>No drone (plain)</span></div>
"""
    )


def _dial(angle: float, theme: dict[str, str]) -> str:
    x, y = 22 + 15 * np.cos(angle - np.pi / 2), 22 + 15 * np.sin(angle - np.pi / 2)
    return (
        f'<svg viewBox="0 0 44 44" width="44" height="44" aria-hidden="true"><circle cx="22" cy="22" r="17" fill="none" stroke="{theme["grid"]}" stroke-width="3"/>'
        f'<line x1="22" y1="22" x2="{x:.1f}" y2="{y:.1f}" stroke="{theme["accent"]}" stroke-width="3" stroke-linecap="round"/>'
        f'<circle cx="22" cy="22" r="3" fill="{theme["accent"]}"/></svg>'
    )


def _quantum_panel(result: dict[str, Any], detector: QuantumDetector, engine: str, theme: dict[str, str]) -> None:
    rows = "".join(
        f'<div class="qk-qrow">{_dial(angle, theme)}<div><span>q{i}: {FEATURE_LABELS.get(name, name)}</span>'
        f'<b>{np.degrees(angle):+.0f}° phase</b></div><code>x = {value:+.2f}</code></div>'
        for i, (name, value, angle) in enumerate(zip(result["names"], result["scaled_mean"], result["angles"]))
    )
    similarity, labels = result["similarity"], detector.train_labels
    top = np.argsort(similarity)[::-1][:5]
    _html(
        """
<div class="qk-section"><h2>What the qubits heard</h2>
<p class="qk-lead">Four features become phase rotations on four entangled qubits. The resulting quantum state is compared with 24 labelled
recordings, and the overlaps decide the vote.</p></div>
"""
    )
    left, right = st.columns([1, 1.25], gap="large")
    left.markdown(
        f'<div class="qk-panel"><h3>Your sound as qubit rotations</h3><p>Average of all windows, scaled the way the model was trained.</p>'
        f'<div class="qk-qrows">{rows}</div></div>',
        unsafe_allow_html=True,
    )
    phases = result["phases"]
    qubits = int(np.log2(len(phases)))
    labels_basis = [f"|{format(i, f'0{qubits}b')}⟩" for i in range(len(phases))]
    fig = go.Figure(go.Bar(x=labels_basis, y=phases, marker=dict(color=theme["accent"], line=dict(width=0)),
                           hovertemplate="%{x}: %{y:+.0f}° phase<extra></extra>"))
    fig.add_hline(y=0, line_color=theme["muted"], line_width=1)
    fig.update_layout(title=dict(text="Quantum fingerprint: the phase each basis state picks up", x=0, font=dict(size=14)))
    fig.update_yaxes(title_text="phase (degrees)", range=[-190, 190], tickvals=[-180, -90, 0, 90, 180])
    right.plotly_chart(_style_fig(fig, theme, 340), width="stretch", config={"displayModeBar": False}, theme=None)

    order = np.argsort(similarity)[::-1]
    fig = go.Figure(go.Bar(
        x=[f"#{rank + 1}" for rank in range(len(order))], y=similarity[order],
        marker=dict(color=[theme["drone"] if labels[i] == 1 else theme["clear"] for i in order], line=dict(width=0),
                    pattern=dict(shape=["/" if labels[i] == 1 else "" for i in order], fgcolor=theme["bg"])),
        customdata=[["drone" if labels[i] == 1 else "no drone"] for i in order],
        hovertemplate="training recording %{x}<br>%{customdata[0]}<br>overlap %{y:.3f}<extra></extra>",
    ))
    source = "measured ibm_kingston response" if engine == "hardware" else "exact simulation"
    fig.update_layout(title=dict(text=f"Overlap with the 24 training recordings ({source})", x=0, font=dict(size=14)))
    fig.update_yaxes(title_text="state overlap |⟨ψ|φ⟩|²", range=[0, 1])
    fig.update_xaxes(title_text="most similar first; striped red = drone, green = no drone")
    st.plotly_chart(_style_fig(fig, theme, 320), width="stretch", config={"displayModeBar": False}, theme=None)
    st.caption(f"{int(labels[top].sum())} of the 5 closest training recordings are drones. This panel shows the average window; the verdict counts every window.")


def _spectrogram(features: dict[str, Any], theme: dict[str, str]) -> go.Figure:
    mel = features["mel"]
    fig = go.Figure(go.Heatmap(
        z=mel, x=np.arange(mel.shape[1]) * features["mel_seconds"], showscale=False,
        colorscale=[[0, theme["bg"]], [0.5, "#1d3557"], [0.8, theme["accent"]], [1, "#fff4ea"]],
        hovertemplate="%{x:.1f} s<br>mel band %{y}<br>%{z:.0f} dB<extra></extra>",
    ))
    fig.update_layout(title=dict(text="Mel spectrogram: where the sound energy sits over time", x=0, font=dict(size=14)))
    fig.update_xaxes(title_text="seconds")
    fig.update_yaxes(title_text="pitch band, low to high")
    return _style_fig(fig, theme, 300)


# --------------------------------------------------------------------------- pages

def _set_clip_from_sample(sample: dict[str, Any]) -> None:
    path = SAMPLES_DIR / sample["file"]
    st.session_state["clip"] = {"name": f"the {sample['title'].lower()} sample", "bytes": path.read_bytes(), "suffix": path.suffix, "truth": sample["truth"]}


def _set_clip_from_upload() -> None:
    uploaded = st.session_state.get("audio_upload")
    if uploaded is None:
        st.session_state.pop("clip", None)
        return
    st.session_state["clip"] = {"name": uploaded.name, "bytes": uploaded.getvalue(), "suffix": Path(uploaded.name).suffix.lower() or ".wav", "truth": None}


def page_detect(data: dict[str, Any], theme: dict[str, str]) -> None:
    _hero(data)
    _how_it_works()
    _html('<div class="qk-section" id="analyze"><h2>Try the detector</h2><p class="qk-lead">Upload your own outdoor recording, or play a '
          'held-out test clip the model never heard during training.</p></div>')

    detector = load_quantum_detector()
    classical = _build_bundle(data["best_svm"], data["selected_features"], "SVM")
    if classical is not None and not classical.model_path.is_file():
        classical = None
    engines = [key for key in ENGINES if (key == "quantum" and detector) or (key == "hardware" and detector and detector.hardware_model is not None)
               or (key == "classical" and classical)]
    if not engines:
        st.warning("No trained models were found. Run the pipeline in the README, then python src/build_live_detector.py.")
        return

    sample_tab, upload_tab = st.tabs([":material/headphones: Try a sample", ":material/upload: Upload a recording"])
    with sample_tab:
        available = [s for s in SAMPLES if (SAMPLES_DIR / s["file"]).is_file()]
        for row_start in range(0, len(available), 2):
            for column, sample in zip(st.columns(2, gap="medium"), available[row_start:row_start + 2]):
                tag = '<span class="qk-tag drone">drone</span>' if sample["truth"] else '<span class="qk-tag clear">no drone</span>'
                if sample.get("hard"):
                    tag = '<span class="qk-tag hard">hard case</span>'
                detail = sample.get("note", f"{sample['source']}, held-out test set")
                column.markdown(
                    f'<div class="qk-tile"><div class="qk-tile-icon">{_mi(sample["icon"])}</div>'
                    f'<div><b>{sample["title"]}</b><span>{detail}</span></div>{tag}</div>',
                    unsafe_allow_html=True,
                )
                column.button(f"Analyze {sample['title'].lower()}", key=f"sample_{sample['id']}", on_click=_set_clip_from_sample,
                              args=(sample,), width="stretch", icon=":material/play_arrow:")
    with upload_tab:
        st.file_uploader("Audio recording (WAV, FLAC, OGG, MP3, M4A or AIFF)", type=["wav", "flac", "ogg", "mp3", "m4a", "aiff", "aif"],
                         key="audio_upload", on_change=_set_clip_from_upload)
        st.caption("The first two minutes are analysed. Outdoor recordings with a few seconds of sound work best.")

    settings_engine, settings_mode = st.columns([1.75, 1], gap="large")
    engine = settings_engine.segmented_control(
        "Engine", options=engines, format_func=lambda key: ENGINES[key], default=engines[0], key="engine",
        help="Quantum simulator: exact simulation of our trained 4-qubit circuit. ibm_kingston response: the same model using the training "
             "kernel measured on IBM's ibm_kingston and its measured response. Classical benchmark: best classical SVM (6 features, 70 recordings).",
    ) or engines[0]
    sensitivity = settings_mode.segmented_control("Sensitivity", options=list(SENSITIVITY), default="Balanced", key="sensitivity") or "Balanced"
    settings_mode.caption(SENSITIVITY[sensitivity][1])
    if engine in {"quantum", "hardware"}:
        mode = detector.spec["metrics"]["hardware" if engine == "hardware" else "ideal"][sensitivity]
        note = (f" Emulated from kernels measured on {detector.spec['hardware']['backend']}. The fully measured hardware run scored "
                f"F1 {detector.spec['hardware']['measured_recording_f1']:.3f}." if engine == "hardware" else "")
        settings_engine.caption(
            f"Held-out test, {detector.spec['test_recordings']} recordings, {sensitivity} mode: F1 {mode['f1']:.3f}. Catches "
            f"{mode['drone_recall']:.0%} of drones and clears {mode['non_drone_recall']:.0%} of other sounds.{note}"
        )
    else:
        settings_engine.caption("Trained on 70 recordings with 6 features, so it is not a like-for-like comparison with the 24-recording quantum model.")

    clip = st.session_state.get("clip")
    if not clip:
        _html(f'<div class="qk-empty">{_mi("graphic_eq")}No recording yet. Pick a sample above or upload your own to see the quantum verdict.</div>')
        return
    st.audio(clip["bytes"], format=f"audio/{clip['suffix'].lstrip('.') or 'wav'}")
    try:
        with st.spinner("Encoding every second into qubits"):
            features = _clip_features(clip["bytes"], clip["suffix"])
            prediction = _predict(engine, features, detector, classical)
    except Exception as exc:
        st.error(f"We could not analyse this recording: {exc}")
        return
    result = {**prediction, "duration": features["duration"]}
    _verdict(result, clip, engine, sensitivity, theme)
    if engine in {"quantum", "hardware"}:
        _quantum_panel(result, detector, engine, theme)
    st.plotly_chart(_spectrogram(features, theme), width="stretch", config={"displayModeBar": False}, theme=None)
    with st.expander("Acoustic features, averaged across windows"):
        _table(pd.DataFrame({"Feature": [FEATURE_LABELS.get(n, n) for n in result["names"]], "Value": [f"{result['means'][n]:.4f}" for n in result["names"]]}))


def _classical_docs() -> None:
    docs = [doc for doc in CLASSICAL_DOCS if (DOCS_DIR / doc["file"]).is_file()]
    if not docs:
        return
    _html('<div class="qk-section"><h2>The classical side, in depth</h2><p class="qk-lead">Four classical models frame the quantum result: '
          'an RBF SVM (A), a small and a larger neural network (B and C) and a spectrogram CNN (D). These write-ups explain the methods and '
          'the protocol for a fair comparison. They describe the study design rather than report new results.</p></div>')
    for column, doc in zip(st.columns(len(docs), gap="medium"), docs):
        column.markdown(
            f'<div class="qk-doc"><div class="qk-tile-icon">{_mi("description")}</div><div><b>{doc["title"]}</b>'
            f'<span class="kind">{doc["kind"]}</span><p>{doc["summary"]}</p></div></div>',
            unsafe_allow_html=True,
        )
        download, github = column.columns(2)
        download.download_button("Download PDF", data=(DOCS_DIR / doc["file"]).read_bytes(), file_name=doc["file"],
                                 mime="application/pdf", icon=":material/download:", width="stretch", key=f"doc_{doc['file']}")
        github.link_button("Open on GitHub", DOCS_URL + doc["file"], icon=":material/open_in_new:", width="stretch")


def page_research(data: dict[str, Any], theme: dict[str, str]) -> None:
    matched = _matched_model_results(data)
    qpu = _qpu_row(data)
    hardware = data.get("hardware", {})
    f1 = {row.Model: float(row.f1) for row in matched.itertuples()} if not matched.empty else {}
    if qpu is not None:
        f1["Real QPU"] = float(qpu["f1"])
    _html(
        """
<div class="qk-page-head"><h1>Can a <em>quantum kernel</em> hear a drone?</h1>
<p class="qk-lead">We trained classical and quantum classifiers on just 24 labelled recordings and tested them on 98 held-out ones, on simulators
and on a real IBM quantum computer.</p></div>
"""
    )
    if not f1:
        st.info("No experiment results found yet. Run the pipeline in the README.")
        return

    metrics = [
        ("", f"{f1.get('Trainable QSVC', float('nan')):.3f}", "Quantum kernel, simulator", "test F1, 4 qubits"),
        ("hl", f"{f1.get('Real QPU', float('nan')):.3f}", f"Quantum kernel on {qpu.get('backend_name', 'IBM hardware') if qpu is not None else 'hardware'}", "test F1, real hardware"),
        ("", f"{hardware.get('predictions_changed', 0)} of {hardware.get('test_recordings', 98)}", "Decisions changed on hardware", "versus the simulator"),
        ("", f"{f1.get('SVM', float('nan')):.3f}", "Best classical model", "RBF SVM, test F1"),
    ]
    _html('<div class="qk-metrics">' + "".join(f'<div class="qk-metric {c}"><span>{label}</span><b>{value}</b><small>{note}</small></div>'
                                              for c, value, label, note in metrics) + "</div>")

    _html('<div class="qk-section"><h2>Every model on the same 98 test recordings</h2><p class="qk-lead">F1 balances catching drones against '
          'false alarms; higher is better. All models trained on the same 24 recordings.</p></div>')
    ordered = sorted(f1.items(), key=lambda item: item[1])
    palette = {"classical": theme["neutral"], "quantum": theme["accent"], "hardware": theme["accent"]}
    fig = go.Figure(go.Bar(
        x=[value for _, value in ordered], y=[{"Real QPU": "Quantum, real hardware"}.get(n, n) for n, _ in ordered], orientation="h",
        marker=dict(color=[palette[MODEL_FAMILY.get(n, "classical")] for n, _ in ordered],
                    pattern=dict(shape=["/" if n == "Real QPU" else "" for n, _ in ordered], fgcolor=theme["bg"]), line=dict(width=0)),
        text=[f"{value:.3f}" for _, value in ordered], textposition="outside", hovertemplate="%{y}: F1 %{x:.3f}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, max(f1.values()) * 1.22], title_text="test F1")
    st.plotly_chart(_style_fig(fig, theme, 320), width="stretch", config={"displayModeBar": False}, theme=None)
    st.caption("Orange bars are quantum models (striped: measured on real hardware); blue-grey bars are classical baselines.")
    with st.expander("Show the numbers as a table"):
        table = matched[["Model", "f1", "balanced_accuracy", "drone_recall", "non_drone_recall"]].copy()
        if qpu is not None:
            table.loc[len(table)] = ["Real QPU", qpu["f1"], qpu["balanced_accuracy"], qpu["drone_recall"], qpu["non_drone_recall"]]
        table.columns = ["Model", "F1", "Balanced accuracy", "Drone recall", "Non-drone recall"]
        _table(table.round(3))

    final = data.get("final", pd.DataFrame())
    ladder = [("Ideal quantum simulator", "Ideal"), ("Finite-shot zero-noise simulator", "Finite shots"),
              ("Stage 6 noisy simulator (2% synthetic)", "2% synthetic noise"), ("Backend-derived noisy simulator", "Chip noise model"),
              ("REAL IBM QPU", "Real hardware")]
    if not final.empty:
        rows = final.set_index("model")
        points = [(label, rows.loc[name]) for name, label in ladder if name in rows.index and pd.notna(rows.loc[name].get("f1"))]
        if points:
            _html('<div class="qk-section"><h2>From perfect simulation to a real chip</h2><p class="qk-lead">The same frozen model, evaluated as the '
                  'simulation gets closer to physical hardware. Every simulator predicted a small effect. The real chip disagreed.</p></div>')
            labels = [label for label, _ in points]
            fig = go.Figure()
            for metric, name, color, width in (("f1", "F1", theme["accent"], 3.5), ("balanced_accuracy", "Balanced accuracy", theme["neutral"], 2.5)):
                values = [float(row[metric]) for _, row in points]
                fig.add_trace(go.Scatter(x=labels, y=values, name=name, mode="lines+markers+text", text=[f"{v:.3f}" for v in values],
                                         textposition="top center", line=dict(color=color, width=width), marker=dict(size=9)))
            fig.add_vrect(x0=len(labels) - 1.5, x1=len(labels) - 0.5, fillcolor=theme["accent"], opacity=0.07, line_width=0)
            fig.update_yaxes(range=[0.3, 0.85], title_text="score")
            st.plotly_chart(_style_fig(fig, theme, 360), width="stretch", config={"displayModeBar": False}, theme=None)

    if qpu is not None and hardware:
        layout = hardware.get("physical_qubits", [])
        chain = '<div class="qk-wire"></div>'.join(f'<div class="qk-qubit">q{q}</div>' for q in layout)
        agreement = hardware["kernel_agreement"]["test"]
        drift = hardware.get("drift")
        drift_html = (f'<p style="margin-top:.8rem">Re-measuring {drift["circuits"]:,} circuits after a chip recalibration gave the same values '
                      f'(correlation {drift["correlation"]:.3f}), so the hardware result is reproducible.</p>' if drift else "")
        _html('<div class="qk-section"><h2>What happened on the chip</h2></div>')
        left, right = st.columns([1, 1.15], gap="large")
        left.markdown(
            f"""
<div class="qk-panel"><span class="qk-hw-tag">{_mi("memory")}{hardware['backend']}, IBM Heron processor</span>
  <div class="qk-chain" aria-label="Physical qubits {', '.join(map(str, layout))} in a line">{chain}</div>
  <div class="qk-facts">
    <div><span>Circuits</span><b>{hardware['circuits']:,}</b></div><div><span>Shots each</span><b>{hardware['shots']}</b></div>
    <div><span>Total shots</span><b>{hardware['circuits'] * hardware['shots'] / 1e6:.2f} M</b></div><div><span>Jobs</span><b>{len(hardware['jobs'])}</b></div>
  </div>
  <p>Each circuit measures how alike two recordings look to the quantum model. The chip kept the pattern (correlation {agreement['correlation']:.2f})
  but squeezed every value toward the middle, about {agreement['slope']:.2f} times the ideal plus {agreement['intercept']:.2f}. That squeeze flipped
  {hardware['predictions_changed']} of {hardware['test_recordings']} test decisions.</p>{drift_html}
</div>""",
            unsafe_allow_html=True,
        )
        sample = hardware.get("test_kernel_sample", {})
        if sample:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sample["ideal"], y=sample["hardware"], mode="markers", name="kernel values",
                                     marker=dict(size=6, color=theme["accent"], opacity=0.5), hovertemplate="simulator %{x:.3f}<br>hardware %{y:.3f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect agreement", line=dict(color=theme["muted"], dash="dash")))
            fig.add_trace(go.Scatter(x=[0, 1], y=[agreement["intercept"], agreement["slope"] + agreement["intercept"]], mode="lines",
                                     name="hardware trend", line=dict(color=theme["text"], width=2.5)))
            fig.update_xaxes(range=[0, 1], title_text="simulator kernel value")
            fig.update_yaxes(range=[0, 1], title_text="hardware kernel value")
            fig.update_layout(title=dict(text="Simulator versus hardware, test kernel", x=0, font=dict(size=14)))
            _style_fig(fig, theme, 450).update_layout(legend=dict(orientation="h", y=-0.22, x=0), margin=dict(b=70))
            right.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, theme=None)

    if {"SVM", "Trainable QSVC"}.issubset(f1):
        model_gap = f1["SVM"] - f1["Trainable QSVC"]
        hw_gap = f1["Trainable QSVC"] - f1.get("Real QPU", f1["Trainable QSVC"])
        total_gap = max(model_gap + hw_gap, 1e-9)
        compression = 1 - hardware["kernel_agreement"]["test"]["slope"] if hardware else None
        _html(
            f"""
<div class="qk-section"><h2>What would it take for quantum to win?</h2>
<p class="qk-lead">Today the classical SVM still leads by {total_gap:.3f} F1 over the real-hardware result. That gap has two parts.</p></div>
<div class="qk-gapbar" role="img" aria-label="Gap of {total_gap:.3f} F1: {model_gap:.3f} from the model, {hw_gap:.3f} from hardware noise">
  <span style="width:{100 * max(model_gap, 0) / total_gap:.1f}%;background:{theme['neutral']}">model {model_gap:+.3f}</span>
  <span style="width:{100 * max(hw_gap, 0) / total_gap:.1f}%;background:{theme['accent']}">hardware noise {hw_gap:+.3f}</span>
</div>
<div class="qk-reqs">
  <div class="qk-req">{_mi("blur_on")}<h3>A better quantum model</h3><p>Even noise-free, the 4-qubit kernel trails the SVM by {model_gap:.3f} F1, and the
  fixed 5- and 6-qubit kernels we tried did no better. It needs a more expressive trainable feature map, not just more qubits.</p></div>
  <div class="qk-req">{_mi("memory")}<h3>Quieter hardware</h3><p>Real noise cost another {hw_gap:.3f} F1{f" by compressing kernel values about {compression:.0%}" if compression is not None else ""}.
  Error mitigation or lower error rates must shrink that compression toward zero.</p></div>
  <div class="qk-req">{_mi("graphic_eq")}<h3>More drone recordings</h3><p>Only 35 drone recordings were available for training. Every model, quantum or
  classical, is starved of labelled drone audio.</p></div>
</div>
"""
        )

    _classical_docs()

    st.markdown("")
    with st.expander("Noise sweep on the simulator"):
        noisy = data.get("noisy", pd.DataFrame())
        if noisy.empty:
            st.info("No saved noise-sweep results are available.")
        else:
            fig = go.Figure()
            for (name, frame), color in zip(noisy.sort_values("error_level").groupby("noise_type"), [theme["accent"], theme["neutral"], theme["text"]]):
                fig.add_trace(go.Scatter(x=frame["error_level"] * 100, y=frame["f1"], name=name, mode="lines+markers", line=dict(color=color, width=3)))
            fig.update_xaxes(title_text="simulated two-qubit error (%)")
            fig.update_yaxes(title_text="test F1", range=[0, 1])
            st.plotly_chart(_style_fig(fig, theme, 320), width="stretch", config={"displayModeBar": False}, theme=None)
    with st.expander("How much training data helps"):
        ideal = data.get("ideal", pd.DataFrame())
        if not ideal.empty:
            sizes = ideal[ideal["evaluation_split"].eq("test") & ideal["experiment"].eq("matched_sample_clean_model_robustness")
                          & ideal["evaluation_snr"].eq("clean") & ideal["feature_count"].eq(4) & ideal["training_snr"].eq("clean")]
            fig = go.Figure()
            for model, label, color, dash in (("rbf_svm", "SVM", theme["neutral"], "solid"), ("small_mlp", "MLP", theme["neutral"], "dot"),
                                              ("fixed_qsvc", "Fixed QSVC", theme["accent"], "dot"), ("trainable_qsvc", "Trainable QSVC", theme["accent"], "solid")):
                frame = sizes[sizes["model"].eq(model)].groupby("training_size_requested", as_index=False)["f1"].mean()
                if not frame.empty:
                    fig.add_trace(go.Scatter(x=frame["training_size_requested"], y=frame["f1"], name=label, mode="lines+markers",
                                             line=dict(color=color, width=3, dash=dash), marker=dict(size=9)))
            fig.update_xaxes(title_text="training recordings")
            fig.update_yaxes(title_text="test F1", range=[0, 1])
            st.plotly_chart(_style_fig(fig, theme, 320), width="stretch", config={"displayModeBar": False}, theme=None)
    with st.expander("Hardware job log"):
        jobs = data.get("qpu_jobs", {}).get("jobs", []) if isinstance(data.get("qpu_jobs"), dict) else []
        if jobs:
            frame = pd.DataFrame(jobs)
            columns = [c for c in ["job_id", "status", "circuit_index_start", "circuit_index_end_exclusive", "circuit_count", "shots"] if c in frame]
            _table(frame[columns])
            st.caption("Read-only snapshot of the local ledger. This page never contacts IBM.")
        else:
            st.info("No hardware jobs are recorded.")


TEAM = ["Sai Srikar Reddy Kolli", "Dhanya Boyapally", "Joseph Johnson"]
PAPERS = {
    "QubitSky write-ups": [
        ("Can a computer hear a drone? A classical machine-learning approach to acoustic drone detection under noise",
         "QubitSky team, accessible research paper, 2026", DOCS_URL + "classical_approach_beginner.pdf",
         "Our plain-language explanation of the four classical baselines and how a fair comparison is designed."),
        ("Classical baselines for acoustic drone detection: Models A to D, evaluation protocol, and CPU/CUDA execution",
         "QubitSky team, technical report, 2026", DOCS_URL + "classical_approach.pdf",
         "The technical methods behind the classical side of QubitSky."),
    ],
    "Quantum kernels": [
        ("Supervised learning with quantum-enhanced feature spaces", "Havlíček et al., Nature, 2019", "https://doi.org/10.1038/s41586-019-0980-2",
         "Introduced quantum kernel classifiers built on entangling feature maps, the idea behind our QSVC."),
        ("Quantum machine learning in feature Hilbert spaces", "Schuld and Killoran, Physical Review Letters, 2019", "https://doi.org/10.1103/PhysRevLett.122.040504",
         "Frames encoding data into quantum states as a kernel method, which is how our model compares recordings."),
        ("Covariant quantum kernels for data with group structure", "Glick et al., Nature Physics, 2024", "https://doi.org/10.1038/s41567-023-02340-9",
         "Introduced quantum kernel alignment, the training approach behind the trainer that tunes our circuit's weights."),
        ("Training quantum embedding kernels on near-term quantum computers", "Hubregtsen et al., Physical Review A, 2022", "https://doi.org/10.1103/PhysRevA.106.042431",
         "Shows how trainable quantum kernels behave on real noisy hardware, the question our ibm_kingston run tests."),
        ("Quantum computing with Qiskit", "Javadi-Abhari et al., arXiv:2405.08810, 2024", "https://arxiv.org/abs/2405.08810",
         "The software stack we used to build, simulate and run our circuits on IBM Quantum hardware."),
    ],
    "Acoustic drone detection and data": [
        ("Empirical study of drone sound detection in real-life environment with deep neural networks", "Jeon et al., EUSIPCO, 2017",
         "https://doi.org/10.23919/EUSIPCO.2017.8081531", "Early evidence that drones can be heard reliably in real outdoor noise."),
        ("Audio based drone detection and identification using deep learning", "Al-Emadi et al., IWCMC, 2019", "https://doi.org/10.1109/IWCMC.2019.8766732",
         "A deep-learning baseline for acoustic drone detection that classical approaches are compared against."),
        ("Real-time drone detection and tracking with visible, thermal and acoustic sensors", "Svanström et al., ICPR, 2021",
         "https://doi.org/10.1109/ICPR48806.2021.9413241", "The study behind our drone and helicopter recordings, including the audio channel."),
        ("A dataset for multi-sensor drone detection", "Svanström et al., Data in Brief, 2021", "https://doi.org/10.1016/j.dib.2021.107521",
         "The Svanström dataset itself: labelled drone, helicopter and background audio."),
        ("ESC: Dataset for environmental sound classification", "Piczak, ACM Multimedia, 2015", "https://doi.org/10.1145/2733373.2806390",
         "ESC-50, the source of our look-alike sounds: birds, insects, wind, rain, aircraft and more."),
    ],
}


def page_about(data: dict[str, Any], theme: dict[str, str]) -> None:
    confusers = ["birdsong", "crows", "insects", "crickets", "wind", "rain", "thunder", "helicopters", "airplanes", "sirens", "fireworks", "footsteps"]
    chips = "".join(f"<span>{c}</span>" for c in confusers)
    stories = [
        ("hearing", "Why sound", "Cameras struggle at night, in fog and behind trees. Microphones are cheap, small and listen in every direction."),
        ("pest_control", "Why it is hard", f"Plenty of everyday sounds hum, buzz or whir like a drone. We trained against all of these.<div class='qk-chips'>{chips}</div>"),
        ("blur_on", "Where quantum fits", "A quantum kernel compares recordings in a space classical computers find hard to simulate. We test whether that helps when labelled data is scarce."),
        ("rule", "Built honestly", "Recordings never leak between training and test, the test set never tunes anything, and every number on this site comes from a recorded run."),
    ]
    team = "".join(
        f'<div class="qk-person"><span class="qk-initials" aria-hidden="true">{name.split()[0][0]}{name.split()[-1][0]}</span>'
        f'<div><b>{name}</b><span>University of Missouri</span></div></div>'
        for name in TEAM
    )
    papers = "".join(
        f'<div class="qk-paper-group"><h3>{group}</h3><ul class="qk-papers">'
        + "".join(
            f'<li><a href="{url}" target="_blank" rel="noopener">{title}</a><span class="cite">{cite}</span><span class="why">{why}</span></li>'
            for title, cite, url, why in items
        )
        + "</ul></div>"
        for group, items in PAPERS.items()
    )
    _html(
        """
<div class="qk-page-head"><h1>Ears for the <em>sky</em></h1>
<p class="qk-lead">QubitSky is a research prototype built for the Qiskit Fall Fest 2026 hackathon at the University of Missouri. Can sound alone reveal a
nearby drone, and can a quantum model help when labelled data is scarce?</p></div>
<div class="qk-stories">"""
        + "".join(f'<div class="qk-story">{_mi(icon)}<h3>{title}</h3><p>{body}</p></div>' for icon, title, body in stories)
        + f"""</div>
<div class="qk-section"><h2>The team</h2><p class="qk-lead">Built for Qiskit Fall Fest 2026.</p></div>
<div class="qk-team">{team}</div>
<div class="qk-section"><h2>Research we built on</h2><p class="qk-lead">The papers and datasets behind QubitSky. Each link opens the publication.</p></div>
{papers}
<div class="qk-section"><h2>Designed for every user</h2></div>
<dl class="qk-dl">
  <dt>Never colour alone</dt><dd>Verdicts use an icon and words. The timeline marks drone seconds with stripes as well as colour, and chart bars repeat the pattern.</dd>
  <dt>Display controls</dt><dd>Dark, light and high-contrast themes, larger text and reduce motion live under Display. Your system's reduced-motion setting is respected too.</dd>
  <dt>Readable by machines</dt><dd>Charts come with data tables, the verdict is announced to screen readers, a skip link jumps to the detector, and everything works with a keyboard.</dd>
  <dt>Adapts to the job</dt><dd>Sensitivity modes tune the same quantum detector for perimeter security, everyday use or busy soundscapes.</dd>
  <dt>Any recording</dt><dd>WAV, FLAC, OGG, MP3, M4A or AIFF, of any length. The model listens one second at a time.</dd>
  <dt>Runs anywhere</dt><dd>CPU only, no GPU needed. The hardware experiment targets different IBM backends with one flag.</dd>
</dl>
<div class="qk-section"><h2>Data and credits</h2>
<p class="qk-lead">Svanström Drone Detection Dataset (CC0). ITU-ARIS Lab Acoustic Drone Dataset (CC BY 4.0). ESC-50 by K. J. Piczak, 2015
(CC BY-NC 3.0, clips from Freesound). NASA Small UAS Flyover Acoustics, used only as an external test. Demo samples are held-out test recordings
from Svanström and ESC-50, and all data stays under its original licence.</p></div>
<div class="qk-footer">QubitSky by Sai Srikar Reddy Kolli, Dhanya Boyapally and Joseph Johnson. Qiskit Fall Fest 2026, University of Missouri.</div>
"""
    )


def main() -> None:
    favicon = ASSETS_DIR / "favicon.png"
    st.set_page_config(page_title="QubitSky: quantum acoustic drone detection", page_icon=str(favicon) if favicon.is_file() else ":satellite:", layout="wide")
    theme = THEMES[THEME_MODES.get(st.session_state.get("theme_mode") or "Dark", "default")]
    _inject_style(theme, bool(st.session_state.get("large_text")), bool(st.session_state.get("calm")))
    data = load_results()

    page = _nav()
    st.session_state["last_page"] = page
    if page == "Detect":
        page_detect(data, theme)
    elif page == "Research":
        page_research(data, theme)
    else:
        page_about(data, theme)



if __name__ == "__main__":
    main()

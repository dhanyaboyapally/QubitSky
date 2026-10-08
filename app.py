from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import SAMPLE_RATE
from extract_features import FEATURE_COLUMNS, extract_features, split_windows


RESULTS_DIR = PROJECT_ROOT / "results"
MAX_ANALYSIS_SECONDS = 120
QUANTUM_DIR = RESULTS_DIR / "quantum"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"


@dataclass
class ModelBundle:
    label: str
    model_path: Path
    scaler_path: Path
    feature_names: list[str]
    feature_count: int
    validation_f1: float


def _inject_style(background_image: Path) -> None:
    if background_image.is_file():
        image_data = base64.b64encode(background_image.read_bytes()).decode("ascii")
        hero_background = (
            "background-image: linear-gradient(rgba(3, 11, 20, 0.50), rgba(3, 11, 20, 0.50)), "
            f"url('data:image/jpeg;base64,{image_data}');"
        )
    else:
        hero_background = "background-image: linear-gradient(135deg, #0a1620, #112a35 55%, #1a3e47);"
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

:root {{
  --ink: #0f2530;
  --muted: #4e6976;
  --surface: #f6f7f2;
  --accent: #ff8a36;
  --accent-alt: #00a8b5;
  --ring: rgba(255,138,54,0.25);
}}

html, body, [class*="stApp"] {{
  font-family: 'Space Grotesk', sans-serif;
  color: var(--ink);
  background: linear-gradient(180deg, #f6f7f2 0%, #f4f5ef 100%);
}}

section[data-testid="stMain"] .block-container {{
    max-width: none;
    padding: 4rem 0 2rem 0;
}}

section[data-testid="stSidebar"], button[data-testid="stExpandSidebarButton"] {{
    display: none;
}}

[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #112531, #0b1a24);
}}

[data-testid="stSidebar"] * {{
  color: #eff7fb;
}}

.qk-hero {{
  position: relative;
    display: flex;
    align-items: flex-start;
    min-height: 42vh;
    width: 100%;
    border-radius: 0;
  {hero_background}
  background-size: cover;
  background-position: center;
    padding: clamp(2.4rem, 7vh, 4.4rem) clamp(1.5rem, 8vw, 8rem) 2rem;
  color: #f6fbff;
  overflow: hidden;
    box-sizing: border-box;
}}

.qk-hero-content {{
    position: relative;
    z-index: 4;
    width: min(740px, 78%);
    margin-top: 1vh;
    text-shadow: 0 2px 18px rgba(0, 0, 0, 0.38);
}}

.qk-hero::after {{
    content: "";
  position: absolute;
    inset: 0;
    z-index: 1;
    pointer-events: none;
    background: linear-gradient(180deg, rgba(3, 11, 20, 0.10), transparent 52%, rgba(3, 11, 20, 0.18));
}}

.qk-hero-title {{
    margin: 0.2rem 0 0.2rem;
    max-width: 12ch;
    font-size: clamp(2.5rem, 5vw, 4.3rem);
    line-height: 1;
    letter-spacing: 0;
}}

.qk-hero-copy {{
    max-width: 58ch;
    color: rgba(245, 251, 255, 0.94);
    font-size: 1rem;
    line-height: 1.5;
    margin: 0;
}}

.qk-hero-cta {{
    display: inline-block;
    margin-top: 0.9rem;
    padding: 0.66rem 1rem;
    border-radius: 8px;
    background: #ff8a36;
    color: #10232c !important;
    font-weight: 700;
    text-decoration: none !important;
    text-shadow: none;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.20);
}}

.qk-radar {{
    position: absolute;
    z-index: 2;
    width: min(28vw, 330px);
    aspect-ratio: 1;
    right: 8%;
    top: 16%;
    border: 1px solid rgba(219, 249, 255, 0.34);
    border-radius: 50%;
    opacity: 0.5;
    pointer-events: none;
}}

.qk-radar-sweep {{
    position: absolute;
    inset: 0;
    border-radius: 50%;
    background: conic-gradient(from 0deg, transparent 0deg 300deg, rgba(115, 238, 226, 0.30) 350deg, transparent 360deg);
    animation: radar-spin 18s linear infinite;
}}

.qk-bird {{
    position: absolute;
    z-index: 3;
    width: 30px;
    height: 12px;
    color: rgba(9, 25, 36, 0.78);
    pointer-events: none;
    animation: bird-glide 24s linear infinite;
}}

.qk-bird::before, .qk-bird::after {{
    content: "";
    position: absolute;
    top: 0;
    width: 12px;
    height: 7px;
    border-top: 2px solid currentColor;
    border-radius: 50%;
}}

.qk-bird::before {{ right: 8px; transform: rotate(18deg); }}
.qk-bird::after {{ left: 8px; transform: rotate(-18deg); }}
.qk-bird-one {{ top: 36%; left: -5%; }}

.qk-drone {{
    position: absolute;
    z-index: 3;
    right: 21%;
    top: 39%;
    width: 44px;
    height: 28px;
    opacity: 0.72;
    pointer-events: none;
    animation: drone-hover 8s ease-in-out infinite;
}}

.qk-drone-body {{
    position: absolute;
    left: 15px;
    top: 10px;
    width: 15px;
    height: 8px;
    border-radius: 4px;
    background: #111e27;
    box-shadow: 0 1px 3px rgba(255,255,255,0.6);
}}

.qk-drone-body::before, .qk-drone-body::after {{
    content: "";
    position: absolute;
    left: -9px;
    top: 3px;
    width: 33px;
    height: 2px;
    background: #172a35;
    transform: rotate(25deg);
}}

.qk-drone-body::after {{ transform: rotate(-25deg); }}

.qk-prop {{
    position: absolute;
    top: 3px;
    width: 12px;
    height: 3px;
    border-radius: 50%;
    background: rgba(12, 31, 43, 0.85);
    opacity: 0.72;
}}

.qk-prop-left {{ left: 0; }}
.qk-prop-right {{ right: 0; }}

.qk-drone-beam {{
    position: absolute;
    top: 21px;
    left: 21px;
    width: 1px;
    height: 42px;
    background: linear-gradient(rgba(149, 255, 229, 0.56), transparent);
}}

@keyframes radar-spin {{ to {{ transform: rotate(360deg); }} }}
@keyframes bird-glide {{ to {{ transform: translateX(120vw) translateY(-18px); }} }}
@keyframes drone-hover {{ 50% {{ transform: translateY(-7px) translateX(5px); }} }}
}}

.qk-kicker {{
  text-transform: uppercase;
  font-family: 'IBM Plex Mono', monospace;
  letter-spacing: 0.14em;
  font-size: 0.76rem;
  color: rgba(247, 253, 255, 0.90);
}}

.qk-title {{
  margin: 0.35rem 0 0.35rem 0;
  font-size: clamp(1.7rem, 4vw, 2.6rem);
  line-height: 1.1;
  letter-spacing: -0.03em;
}}

.qk-sub {{
  max-width: 70ch;
  color: rgba(241, 249, 255, 0.92);
  margin-bottom: 0;
}}

.qk-card {{
  background: linear-gradient(160deg, #ffffff, #f9f8f3);
  border: 1px solid #dde7ea;
  border-radius: 16px;
  padding: 1rem 1rem 0.9rem 1rem;
  box-shadow: 0 10px 28px rgba(16, 34, 43, 0.08);
}}

.qk-result {{
    border-left: 6px solid var(--accent-alt);
    background: #eaf7f5;
    padding: 1.25rem 1.4rem;
    margin: 1rem 0;
    color: #102833;
}}

.qk-result.drone {{ border-color: #d24b36; background: #fff0eb; }}
.qk-result-title {{ font-size: clamp(1.8rem, 5vw, 3rem); line-height: 1.05; font-weight: 700; }}

.qk-stat {{
  font-size: 1.5rem;
  font-weight: 700;
  color: #102833;
}}

.qk-muted {{
  color: var(--muted);
  font-size: 0.93rem;
}}

.qk-chip {{
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.76rem;
  border-radius: 999px;
  padding: 0.22rem 0.6rem;
  background: #e6f8fb;
  color: #10566f;
  border: 1px solid #bcecf2;
  margin-right: 0.45rem;
  margin-bottom: 0.35rem;
}}

.qk-alert {{
  border-left: 4px solid var(--accent);
  background: #fff4ea;
  border-radius: 10px;
  padding: 0.75rem 0.9rem;
  color: #6b3c17;
}}

.stTabs [data-baseweb="tab-list"] {{
  gap: 0.4rem;
}}

.stTabs [data-baseweb="tab"] {{
  border-radius: 999px;
  padding: 0.35rem 0.9rem;
}}

.stButton button, .stDownloadButton button {{
  border-radius: 10px;
  border: 1px solid #d6dde0;
  background: #ffffff;
}}

.stButton button:hover, .stDownloadButton button:hover {{
  border-color: #ffb487;
  box-shadow: 0 0 0 0.2rem var(--ring);
}}

@keyframes fade-slide {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}

@media (max-width: 900px) {{
    .qk-hero {{ min-height: 37vh; padding: 2.8rem 1.25rem 1.6rem; background-position: center; }}
    .qk-hero-content {{ width: 100%; margin-top: 0; }}
    .qk-hero-title {{ max-width: 11ch; font-size: clamp(2.5rem, 10vw, 4rem); }}
    .qk-radar {{ width: 34vw; right: 5%; top: 48%; }}
    .qk-drone {{ right: 16%; top: 60%; }}
}}

@media (prefers-reduced-motion: reduce) {{
    .qk-bird, .qk-radar-sweep, .qk-drone {{ animation: none !important; }}
    * {{ transition: none !important; }}
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


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
        "svm": _safe_read_csv(RESULTS_DIR / "svm_results.csv"),
        "mlp": _safe_read_csv(RESULTS_DIR / "mlp_results.csv"),
        "qpu_jobs": _safe_read_json(QUANTUM_DIR / "qpu_jobs.json"),
        "selected_features": _safe_read_json(RESULTS_DIR / "selected_features.json"),
        "best_svm": _safe_read_json(RESULTS_DIR / "best_svm_configuration.json"),
        "best_mlp": _safe_read_json(RESULTS_DIR / "best_mlp_configuration.json"),
        "stage7": _safe_read_json(MODELS_DIR / "quantum" / "stage7_frozen_config.json"),
        "qpu_results": _safe_read_csv(QUANTUM_DIR / "qpu_results.csv"),
    }


@st.cache_resource(show_spinner=False)
def _load_model(model_path: str) -> Any:
    return joblib.load(model_path)


@st.cache_resource(show_spinner=False)
def _load_scaler(scaler_path: str) -> Any:
    return joblib.load(scaler_path)


def _extract_audio_bundle(audio_file, extension: str) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Load the upload and extract features per window, exactly as training does."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
        tmp.write(audio_file.getbuffer())
        tmp_path = Path(tmp.name)
    try:
        audio, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True, duration=MAX_ANALYSIS_SECONDS)
        audio = audio.astype(np.float32, copy=False)
        return audio, [extract_features(window) for window in split_windows(audio)]
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _build_bundle(config: dict[str, Any], selected_features: dict[str, Any], label: str) -> ModelBundle | None:
    best = config.get("best_external_model", {})
    model_path = best.get("model_path")
    feature_count = int(best.get("feature_count", 0) or 0)
    if not model_path or feature_count <= 0:
        return None
    mapping = selected_features.get("qubit_mappings", {}).get(str(feature_count), {})
    names = mapping.get("features", [])
    scaler_path = MODELS_DIR / f"scaler_{feature_count}.pkl"
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
        scaler_path=scaler_path,
        feature_names=list(names),
        feature_count=feature_count,
        validation_f1=float(best.get("validation_f1", 0.0) or 0.0),
    )


def _hero() -> None:
    st.markdown(
        """
<section class="qk-hero">
  <div class="qk-radar"><span class="qk-radar-sweep"></span></div>
  <span class="qk-bird qk-bird-one" aria-hidden="true"></span>
  <div class="qk-drone" aria-hidden="true"><span class="qk-prop qk-prop-left"></span><span class="qk-prop qk-prop-right"></span><span class="qk-drone-body"></span><span class="qk-drone-beam"></span></div>
  <div class="qk-hero-content">
    <div class="qk-kicker">Audio-Based Drone Detection</div>
    <h1 class="qk-hero-title">QubitSky</h1>
    <div class="qk-kicker">Listen Beyond the Noise.</div>
    <p class="qk-hero-copy">Upload an outdoor audio recording and QubitSky will analyze its acoustic features to determine whether a drone is present.</p>
    <a class="qk-hero-cta" href="#audio-upload">Upload Audio</a>
  </div>
</section>
        """,
        unsafe_allow_html=True,
    )


def page_detect(data: dict[str, Any]) -> None:
    _hero()
    st.markdown('<div id="audio-upload"></div>', unsafe_allow_html=True)
    st.subheader("Upload Audio")
    uploaded = st.file_uploader(
        "Choose an audio recording",
        type=["wav", "flac", "ogg", "mp3", "m4a", "aiff", "aif"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        key="audio_upload",
    )
    st.session_state["audio_uploaded"] = uploaded is not None
    if uploaded is None:
        st.caption("Supported formats: WAV, FLAC, OGG, MP3, M4A, AIFF")
        with st.expander("How does this work?"):
            st.write("Audio → Acoustic Features → ML / Quantum Model → Drone or No Drone")
            st.write("The quantum model encodes four selected acoustic features into four qubits and compares their quantum-state similarity.")
        return

    st.audio(uploaded)
    model_options = {
        "Classical SVM": _build_bundle(data["best_svm"], data["selected_features"], "SVM"),
        "Small MLP": _build_bundle(data["best_mlp"], data["selected_features"], "MLP"),
    }
    available = {name: bundle for name, bundle in model_options.items() if bundle is not None}
    if not available:
        st.error("The saved local detection models could not be loaded.")
        return
    default_model = "Classical SVM" if "Classical SVM" in available else next(iter(available))
    with st.expander("Advanced model selection"):
        st.caption("Choose an existing local classical model; quantum models are compared on the Research page.")
        model_name = st.selectbox(
            "Detection model",
            options=list(available),
            index=list(available).index(default_model),
            label_visibility="collapsed",
        )
    bundle = available[model_name]
    upload_key = hashlib.sha256(uploaded.getvalue()).hexdigest()

    if st.button("ANALYZE AUDIO", type="primary", width="stretch"):
        try:
            with st.spinner("Analyzing recording…"):
                clip, window_features = _extract_audio_bundle(uploaded, Path(uploaded.name).suffix.lower() or ".wav")
                if not window_features:
                    raise ValueError("The recording contains no audio.")
                feature_frame = pd.DataFrame(
                    [[features[name] for name in bundle.feature_names] for features in window_features],
                    columns=bundle.feature_names,
                )
                model = _load_model(str(bundle.model_path))
                scaler = _load_scaler(str(bundle.scaler_path))
                scaled_features = scaler.transform(feature_frame.to_numpy(dtype=float))
                window_predictions = model.predict(scaled_features).astype(int)
                drone_windows = int(window_predictions.sum())
                # Majority vote over one-second windows.
                prediction = int(drone_windows * 2 >= len(window_predictions))
                st.session_state["analysis_result"] = {
                    "upload_key": upload_key,
                    "model_name": bundle.label,
                    "prediction": prediction,
                    "drone_windows": drone_windows,
                    "window_count": len(window_predictions),
                    "clip": clip,
                    "features": feature_frame.mean().to_dict(),
                    "feature_names": bundle.feature_names,
                    "feature_count": bundle.feature_count,
                    "model_path": str(bundle.model_path),
                }
        except Exception as exc:
            st.error(f"Audio analysis failed: {exc}")

    result = st.session_state.get("analysis_result", {})
    if result.get("upload_key") == upload_key and result.get("model_name") == bundle.label:
        is_drone = result["prediction"] == 1
        result_text = "DRONE DETECTED" if is_drone else "NO DRONE DETECTED"
        result_class = "drone" if is_drone else ""
        st.markdown(
            f"<div class='qk-result {result_class}'><div class='qk-muted'>Prediction</div><div class='qk-result-title'>{result_text}</div><div class='qk-muted'>Model: {result['model_name']} · {result['drone_windows']} of {result['window_count']} one-second windows sounded like a drone</div></div>",
            unsafe_allow_html=True,
        )
        with st.expander("View analysis details"):
            clip = result["clip"]
            st.markdown("**Waveform**")
            st.line_chart(pd.DataFrame({"Amplitude": clip[::80]}), height=180)
            st.markdown("**Spectrogram**")
            spectrum = librosa.amplitude_to_db(np.abs(librosa.stft(clip)), ref=np.max)
            fig = px.imshow(spectrum, origin="lower", aspect="auto", color_continuous_scale="Tealgrn")
            fig.update_layout(height=300, margin=dict(l=8, r=8, t=10, b=8))
            st.plotly_chart(fig, width="stretch")
            feature_table = pd.DataFrame(
                {"Feature": result["feature_names"], "Mean across windows": [result["features"][name] for name in result["feature_names"]]}
            )
            st.dataframe(feature_table, width="stretch", hide_index=True)
            st.caption(f"Local {result['model_name']} · {result['feature_count']} selected features · {Path(result['model_path']).name}")

    with st.expander("How does this work?"):
        st.write("Audio → Acoustic Features → ML / Quantum Model → Drone or No Drone")
        st.write("The quantum model encodes four selected acoustic features into four qubits and compares their quantum-state similarity.")


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
    names = {
        "rbf_svm": "SVM",
        "small_mlp": "MLP",
        "fixed_qsvc": "Fixed QSVC",
        "trainable_qsvc": "Trainable QSVC",
    }
    matched["Model"] = matched["model"].map(names)
    return matched[matched["Model"].notna()]


def page_research(data: dict[str, Any]) -> None:
    st.title("THE RESEARCH BEHIND QUBITSKY")
    st.write(
        "Can quantum-kernel machine learning remain competitive with classical models when labeled data are limited and acoustic conditions are noisy?"
    )

    matched = _matched_model_results(data)
    qpu_results = data.get("qpu_results", pd.DataFrame())
    qpu_final = pd.DataFrame()
    if not qpu_results.empty and {"status", "f1"}.issubset(qpu_results.columns):
        qpu_final = qpu_results[
            # run_ibm_qpu.py writes "completed" once all batches are assembled.
            qpu_results["status"].astype(str).str.lower().isin({"measured", "completed"})
            & qpu_results["f1"].notna()
        ].copy()

    st.subheader("F1 Score by Model")
    chart_rows = matched[["Model", "f1"]].rename(columns={"f1": "F1"}) if not matched.empty else pd.DataFrame()
    if not qpu_final.empty:
        qpu_row = qpu_final.iloc[-1]
        chart_rows = pd.concat(
            [chart_rows, pd.DataFrame([{"Model": "Real QPU", "F1": qpu_row["f1"]}])],
            ignore_index=True,
        )
    if chart_rows.empty:
        st.info("No matched comparison results are available yet.")
    else:
        fig = px.bar(
            chart_rows,
            x="Model",
            y="F1",
            color="Model",
            text_auto=".3f",
            color_discrete_sequence=["#176b70", "#df7834", "#557a46", "#397b9a", "#9c554b"],
        )
        fig.update_layout(height=360, margin=dict(l=8, r=8, t=16, b=8), showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")

    if not matched.empty:
        comparison = matched[["Model", "f1", "balanced_accuracy", "drone_recall"]].rename(
            columns={
                "f1": "F1",
                "balanced_accuracy": "Balanced accuracy",
                "drone_recall": "Drone recall",
            }
        )
        if not qpu_final.empty:
            qpu_row = qpu_final.iloc[-1]
            comparison.loc[len(comparison)] = [
                "Real QPU",
                qpu_row.get("f1"),
                qpu_row.get("balanced_accuracy"),
                qpu_row.get("drone_recall"),
            ]
        st.dataframe(comparison, width="stretch", hide_index=True, column_config={
            "F1": st.column_config.NumberColumn(format="%.3f"),
            "Balanced accuracy": st.column_config.NumberColumn(format="%.3f"),
            "Drone recall": st.column_config.NumberColumn(format="%.3f"),
        })
        st.caption("Matched held-out test set: four selected features and 24 training recordings. Scores are experiment results, not a claim of quantum advantage.")

    st.subheader("Real IBM Quantum Hardware")
    ledger = data.get("qpu_jobs", {}) if isinstance(data.get("qpu_jobs"), dict) else {}
    jobs = ledger.get("jobs", []) if isinstance(ledger.get("jobs", []), list) else []
    completed = sum(str(job.get("status", "")).lower() == "completed" for job in jobs)
    circuit_count = int(ledger.get("total_circuits", 0) or 0)
    per_job = int(ledger.get("circuits_per_job_target", 1) or 1)
    planned = (circuit_count + per_job - 1) // per_job if per_job else 0
    if not qpu_final.empty:
        qpu_row = qpu_final.iloc[-1]
        st.markdown("**REAL QPU RESULT**")
        q1, q2, q3 = st.columns(3)
        q1.metric("F1", f"{float(qpu_row['f1']):.3f}")
        q2.metric("Balanced accuracy", f"{float(qpu_row['balanced_accuracy']):.3f}")
        q3.metric("Drone recall", f"{float(qpu_row['drone_recall']):.3f}")
        st.caption(
            f"Measured on {qpu_row.get('backend_name', 'IBM hardware')}, physical qubits "
            f"{qpu_row.get('physical_qubits', '')}, {int(qpu_row.get('shots', 0))} shots, "
            f"{int(qpu_row.get('circuit_count', 0))} circuits in {int(qpu_row.get('job_count', 0))} jobs."
        )
    else:
        st.markdown(
            f"<div class='qk-card'><strong>Experiment in progress</strong><br>Backend: {ledger.get('backend', 'ibm_pittsburgh')}<br>{completed} / {planned} hardware batches completed</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Noise Robustness"):
        noisy = data.get("noisy", pd.DataFrame())
        if noisy.empty:
            st.info("No saved noise-sweep results are available.")
        else:
            fig = px.line(
                noisy.sort_values("error_level"),
                x="error_level",
                y="f1",
                color="noise_type",
                markers=True,
                labels={"error_level": "Simulated noise level", "f1": "F1 score", "noise_type": "Noise setting"},
                color_discrete_sequence=["#df7834", "#176b70", "#557a46"],
            )
            fig.update_layout(height=330, margin=dict(l=8, r=8, t=12, b=8), yaxis_range=[0, 1])
            st.plotly_chart(fig, width="stretch")

    with st.expander("Training Size"):
        if matched.empty:
            st.info("No matched training-size results are available.")
        else:
            size_results = data["ideal"]
            size_results = size_results[
                size_results["evaluation_split"].eq("test")
                & size_results["experiment"].eq("matched_sample_clean_model_robustness")
                & size_results["evaluation_snr"].eq("clean")
                & size_results["feature_count"].eq(4)
                & size_results["training_snr"].eq("clean")
                & size_results["model"].isin(["rbf_svm", "small_mlp", "fixed_qsvc", "trainable_qsvc"])
            ].copy()
            size_results["Model"] = size_results["model"].map(
                {"rbf_svm": "SVM", "small_mlp": "MLP", "fixed_qsvc": "Fixed QSVC", "trainable_qsvc": "Trainable QSVC"}
            )
            fig = px.line(
                size_results.groupby(["training_size_requested", "Model"], as_index=False)["f1"].mean(),
                x="training_size_requested",
                y="f1",
                color="Model",
                markers=True,
                labels={"training_size_requested": "Training recordings", "f1": "F1 score"},
            )
            fig.update_layout(height=330, margin=dict(l=8, r=8, t=12, b=8), yaxis_range=[0, 1])
            st.plotly_chart(fig, width="stretch")

    with st.expander("View hardware details"):
        st.caption("Read-only local ledger snapshot. This page never contacts IBM or changes job state.")
        if jobs:
            jobs_df = pd.DataFrame(jobs)
            columns = [
                name
                for name in ["job_id", "status", "circuit_index_start", "circuit_index_end_exclusive", "circuit_count", "shots"]
                if name in jobs_df.columns
            ]
            st.dataframe(jobs_df[columns], width="stretch", hide_index=True)
        else:
            st.info("No local hardware jobs are recorded.")


def page_about(data: dict[str, Any]) -> None:
    st.title("ABOUT QUBITSKY")
    st.subheader("WHAT IS QUBITSKY?")
    st.write("QubitSky is an acoustic drone-detection research prototype.")
    st.subheader("WHY SOUND?")
    st.write("Microphones can help detect drones even when cameras have limited visibility.")
    st.subheader("WHAT MAKES THE PROBLEM HARD?")
    st.write("Birds, wind, aircraft, engines, helicopters and insects can sound similar or interfere with detection.")
    st.subheader("WHERE DOES QUANTUM FIT?")
    st.write("QubitSky compares classical machine learning with quantum-kernel classification for small-data and noisy environments.")


def main() -> None:
    st.set_page_config(page_title="QubitSky", page_icon=":satellite:", layout="wide")
    _inject_style(PROJECT_ROOT / "background.jpg")
    data = load_results()
    ambience_path = ASSETS_DIR / "ambience.mp3"

    nav_col, sound_col = st.columns([5, 1], vertical_alignment="center")
    page = nav_col.radio(
        "Navigate",
        ["DETECT", "RESEARCH", "ABOUT"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="main_navigation",
    )
    st.session_state.setdefault("ambient_on", False)
    ambient_label = "🔊" if st.session_state["ambient_on"] else "🔇"
    if sound_col.button(
        ambient_label,
        help="Ambient sound. Optional and off by default; turns off when an audio clip is uploaded.",
        disabled=not ambience_path.is_file(),
    ):
        st.session_state["ambient_on"] = not st.session_state["ambient_on"]

    if page == "DETECT":
        page_detect(data)
    elif page == "RESEARCH":
        page_research(data)
    else:
        page_about(data)

    if st.session_state.get("audio_uploaded", False):
        st.session_state["ambient_on"] = False
    if st.session_state["ambient_on"] and ambience_path.is_file():
        with ambience_path.open("rb") as audio_handle:
            st.audio(audio_handle.read(), format="audio/mp3", autoplay=True, loop=True)


if __name__ == "__main__":
    main()

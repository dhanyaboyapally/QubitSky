# QubitSky

**Quantum acoustic drone detection under noise and data scarcity**

QubitSky listens to a short audio clip and decides whether a drone is in it. The detector is a 4-qubit trainable quantum kernel classifier (QSVC) built with Qiskit, and it was run on real IBM quantum hardware (`ibm_kingston`). We compare it fairly against classical models on the same small, noisy data.

Built for Qiskit Fall Fest 2026 at the University of Missouri, Challenge 2: Quantum Machine Learning.

## What it does

- **Detect:** upload a recording or try one of the samples. The app cuts the audio into 1-second windows and runs the trained quantum kernel on each window. It then reports the share of windows that sound like a drone. You can switch between the ideal simulator, an `ibm_kingston` hardware response, and a classical benchmark.
- **Research:** this page holds the results from every stage. It covers classical against quantum at equal data, the noise sweeps, and the real hardware run.
- **About:** this page has the team, the method, the write-ups and the references.

## Results

All models use the same 4 audio features and the same 24 balanced training recordings. They are scored on the same 98 held-out test recordings (16 drone, 82 non-drone). Splits are made by recording, so no recording appears in more than one split.

| Model | F1 | Balanced accuracy | ROC-AUC | Drones caught | False alarms |
|---|---|---|---|---|---|
| RBF SVM | 0.492 | 0.786 | 0.836 | 15 / 16 | 30 / 82 |
| Small MLP | 0.409 | 0.665 | 0.778 | 9 / 16 | 19 / 82 |
| Fixed quantum kernel (QSVC) | 0.308 | 0.579 | 0.585 | 8 / 16 | 28 / 82 |
| Trainable quantum kernel, ideal simulator | 0.444 | 0.736 | 0.833 | 14 / 16 | 33 / 82 |
| Trainable quantum kernel, real `ibm_kingston` | 0.394 | 0.688 | 0.731 | 14 / 16 | 41 / 82 |

**About the hardware run**

- **Setup:** `ibm_kingston` (Heron r2), physical qubits [42, 43, 44, 45], 512 shots, 4,548 circuits in 4 jobs.
- **Kernel agreement:** the hardware kernel correlates 0.89 with the ideal kernel on the test set, but its values are compressed.
- **Changed predictions:** 20 of 98 test predictions changed.
- **Drone recall:** stayed at 14 of 16.
- **Repeat check:** we ran one batch again after IBM recalibrated the device. The two runs agree at a correlation of 0.985.

The classical RBF SVM still scores best at this data size. We do not claim a quantum advantage. The project measures how a quantum kernel behaves, both in simulation and on real hardware, compared with classical models under the same conditions.

## How it works

1. **Audio:** each recording is resampled to 16 kHz mono and cut into 1-second windows. Short tails are dropped, not padded.
2. **Features:** four features are chosen on training data only and scaled with a scaler fitted on training data only:
   - spectral bandwidth
   - MFCC 3
   - spectral flatness
   - RMS energy
3. **Encoding:** each feature drives one qubit. The circuit has these layers:
   - a Hadamard layer;
   - a trainable phase rotation on each qubit;
   - an entangling phase between neighbouring qubits.

   The kernel value for two clips is the overlap (fidelity) of their quantum states. The circuit's parameters are trained with SPSA over 10 seeded restarts, and the run with the best validation F1 is kept.
4. **Classify:** an SVM uses the kernel values against the 24 training recordings to classify each window. The windows then vote on the whole clip.

The live app evaluates this circuit with an exact closed-form statevector (`src/live_quantum.py`). That version matches Qiskit to within 1e-15, so detection runs fast and needs no IBM account.

## Quick start

Python 3.10 or newer. CPU only; no GPU needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The trained models and results are committed, so the app works without downloading any datasets. The repo also includes a dev container for GitHub Codespaces.

## Reproducing the pipeline

Place the datasets in `data/raw/` (see [Data](#data)), then run:

```bash
# 1. Inventory, recording-level splits, 1 s windows, features, noise variants
python src/build_master_metadata.py
python src/prepare_dataset.py

# 2. Classical baselines
python src/train_svm.py
python src/train_mlp.py

# 3. Quantum kernels on the ideal simulator (fixed and trainable)
python src/quantum_kernel_simulator.py

# 4. Noisy simulation sweep with Aer; also writes the frozen hardware config
python src/quantum_noise_experiments.py --shots 1024 --seed 42

# 5. Real IBM hardware: preflight first (submits nothing), then submit once per batch
python src/run_ibm_qpu.py --backend ibm_kingston
RUN_REAL_QPU=YES python src/run_ibm_qpu.py --backend ibm_kingston --confirm-real-qpu --submit-after-review --shots 512

# 6. Hardware analysis and the live detector used by the app
python src/analyze_hardware.py
python src/build_live_detector.py
```

Step 5 needs IBM Quantum credentials in `secrets/ibm_quantum.env`:

```text
QISKIT_IBM_TOKEN=...
QISKIT_IBM_INSTANCE=...
```

This file is gitignored and must never be committed.

The hardware script can also resume interrupted runs. It refuses to continue if the device calibration drifts by more than 25%. Raw counts for every job are saved in `results/quantum/qpu_job_cache/`.

## Data

| Dataset | Use | Licence |
|---|---|---|
| [Svanström drone detection dataset](https://zenodo.org/records/5500576) | drones, helicopters, background | CC0 |
| [ITU-ARIS acoustic drone dataset](https://huggingface.co/datasets/imm61/itu-aris-lab-acoustic-drone-dataset) | drones and background | CC BY 4.0 |
| [ESC-50](https://github.com/karolpiczak/ESC-50) | confusers: aircraft, birds, insects, weather, sirens and more | CC BY-NC 3.0 |
| [NASA small UAS flyover acoustics](https://data.nasa.gov/dataset/small-uas-flyover-acoustics-data) | external test only, never used for training | public |

Expected layout:

```text
data/raw/
  svanstrom/               # Data/Audio/*.wav
  ddl/                     # ITU-ARIS audio/, groups.json, splits/
  esc50/                   # audio/*.wav, meta/esc50.csv
  nasa_external/           # small_uav_acoustics.zip
```

The split is 301 train, 80 validation and 98 test recordings. Noisy copies at 20, 10, 5 and 0 dB SNR stay in the same split as the recording they came from. See [QubitSky_DATASETS.md](QubitSky_DATASETS.md) for more detail.

## Repository layout

```text
app.py                 Streamlit website (Detect, Research, About)
src/                   pipeline scripts, models, and IBM hardware runner
models/                trained scalers, classical models, quantum detector
results/               metrics, kernels, hardware job cache, plots
docs/                  classical-approach write-ups (PDF and LaTeX)
assets/                favicon and sample clips for the website
```

## Documents

- [docs/classical_approach_beginner.pdf](docs/classical_approach_beginner.pdf): *Can a Computer Hear a Drone?*, an accessible introduction.
- [docs/classical_approach.pdf](docs/classical_approach.pdf): the technical report on classical baselines A to D and how to compare them fairly with the quantum model.

## Team

- Sai Srikar Reddy Kolli
- Dhanya Boyapally
- Joseph Johnson

## Licence

Code is released under the [MIT Licence](LICENSE). The datasets keep their original licences, listed above.

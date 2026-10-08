# QubitSky

**Quantum Acoustic Drone Detection Under Noise and Data Scarcity**

QubitSky is a CPU-first research project for the Qiskit Fall Fest 2026 Challenge 2. It studies whether a quantum-kernel classifier behaves differently from classical models when drone recordings must be distinguished from acoustically similar confusers with limited labeled data and environmental noise.

## Changes on the `srikar` branch

This branch builds on `dhanya` and fixes three problems found while reproducing the pipeline from the raw datasets. The recording-level split is unchanged (train/validation/test = 301/80/98 recordings; the matched quantum subset is 24 balanced recordings). Stage numbers quoted further down this README come from the earlier 3-second pipeline and are superseded by a rerun.

1. **One-second windows, no zero-padding.** ITU-ARIS clips are 1 s long and were padded to 3 s, while ITU-ARIS supplies most drone recordings. The padding made the dataset identifiable from the features: the rule "clip is from ITU-ARIS" alone scored test F1 0.690, and the selected MLP flagged every ITU-ARIS background clip as a drone while missing every Svanström drone. Every source is now cut into 1 s windows (`CLIP_DURATION_SECONDS = 1.0`, `extract_features.split_windows`), and short tails are dropped instead of padded. Feature selection now picks `spectral_bandwidth_mean`, `mfcc_3_mean`, `spectral_flatness_mean` and `rms_energy_mean`.
2. **Reproducible trainable kernel.** SPSA was unseeded, so the trainable QSVC's test F1 ranged 0.43 to 0.71 across reruns. Stage 5 now runs seeded restarts (`--trainable-restarts`, default 10) and keeps the one with the best validation F1; the test split never influences the choice.
3. **Correct classical comparators.** `_fit_classical_comparators` labelled every SVM/MLP row as 4 features and 24 recordings, so later grid configurations overwrote the matched rows used by Stage 6 and the RESEARCH page.

4. **MLP early stopping.** With 24 training recordings an epoch is only 2 optimizer updates, so the 15-epoch patience stopped the MLP at epoch 1 and it predicted "no drone" for every recording (F1 0.000). Early stopping now requires at least 500 updates before it may stop; the best validation checkpoint is still restored.
5. **Stage 5 results key.** The fixed-QSVC matched row shared its key with the noise-robustness rows and was overwritten; `experiment` is now part of the key.

The DETECT page now classifies every 1 s window of an upload and takes a majority vote, and resolves model paths inside the checkout.

Results after the fixes (clean test, recording level, same samples for every model; 4 features, 24 recordings): RBF SVM F1 0.492, trainable QSVC 0.444, MLP 0.409, fixed QSVC 0.308. With the selected 6-feature models, DETECT recognises 83% of held-out Svanström drones (previously 50% for the SVM and 0% for the MLP).

### Real IBM hardware (`ibm_kingston`)

The frozen 4-qubit trainable model ran on `ibm_kingston` (Heron r2, physical qubits [42, 43, 44, 45], 512 shots, 4,548 circuits in 4 jobs, one calibration window, about 628 s of QPU time):

| 98 test recordings | Accuracy | F1 | Balanced acc. | ROC-AUC | False alarms |
|---|---|---|---|---|---|
| Ideal simulator | 0.643 | 0.444 | 0.736 | 0.833 | 33 / 82 |
| Aer with Kingston calibration | 0.663 | 0.459 | 0.748 | | 31 / 82 |
| **Real `ibm_kingston`** | **0.561** | **0.394** | **0.688** | **0.731** | **41 / 82** |

Hardware kernel values correlate about 0.90 with the ideal kernel but are compressed (about 0.83x ideal + 0.05), and 20 of 98 test predictions changed, while the calibration-based noise model predicted no change. Drone recall stayed at 14/16. A first batch 1 measured before an 18:13 recalibration is kept in `results/quantum/qpu_archive/`; it agrees with the rerun at correlation 0.985. Raw counts per job are in `results/quantum/qpu_job_cache/`.

`run_ibm_qpu.py` now takes `--backend` (`ibm_pittsburgh` default, `ibm_kingston`, `ibm_miami`) with per-backend layouts and calibration references, estimates QPU time including the repetition delay (the old estimate was about 70x too low), and offers `--skip-validation-kernel` and 256 shots. Example: `RUN_REAL_QPU=YES python src/run_ibm_qpu.py --backend ibm_kingston --confirm-real-qpu --submit-after-review --shots 512`, run once per batch.

To regenerate everything: place the four datasets in `data/raw/` (see Data below), then run `build_master_metadata.py`, `prepare_dataset.py`, `train_svm.py`, `train_mlp.py`, `quantum_kernel_simulator.py` and `quantum_noise_experiments.py`. Add `--store-noisy-audio` to `prepare_dataset.py` if a model needs the noisy audio itself rather than its features.

## Research Question

Using only 4–6 audio features, how does a quantum-kernel classifier compare with a classical RBF SVM and a small MLP as the training set shrinks and signal-to-noise ratio decreases? What qubit count and simulated error rate, if any, are needed for the quantum model to match the classical baselines?

The working hypothesis is that quantum feature spaces may behave differently when labeled training data is scarce and noise is high. This is a hypothesis to test, not a claim of quantum advantage. Results will come only from recorded experiments; no scores are fabricated or plotted as measured outcomes before runs exist.

## Planned Models And Experiments

- Classical RBF SVM with a small search over `C` and `gamma`.
- Small CPU-friendly MLP with early stopping.
- Qiskit Machine Learning quantum-kernel classifier, using shallow feature maps with 4, 5, or 6 qubits.
- Training subsets near 25, 50, 100, and 200 samples with shared, stratified test sets and recording-level separation.
- Clean audio and 20, 10, 5, and 0 dB SNR conditions.
- Ideal and noisy simulation, including a sweep of simulated error levels before any hardware experiment.
- A separately prepared IBM QPU experiment, only after the full local pipeline works and the user explicitly says `RUN REAL QPU`.

The pipeline will use 4–6 selected features in every final model. Candidate ranking and scaling must be fit on training data only to avoid leakage. Original recordings and their derived noisy versions must stay in the same split.

## Data

Stage 3 sources:

- Svanström Multi-Sensor Drone Detection Dataset: https://zenodo.org/records/5500576
- ESC-50: https://github.com/karolpiczak/ESC-50
- NASA Small UAS Flyover Acoustics: https://data.nasa.gov/dataset/small-uas-flyover-acoustics-data
- ITU ARIS Lab Acoustic Drone Dataset (compact DDL source): https://huggingface.co/datasets/imm61/itu-aris-lab-acoustic-drone-dataset
- Alternative DDL archive (12.6 GB): https://zenodo.org/records/6459183
- Optional UrbanSound8K: https://urbansounddataset.weebly.com/urbansound8k.html

The expected raw-data layout is:

```text
data/raw/
  svanstrom/               # extracted Data/Audio/*.wav
  esc50/
    audio/*.wav
    meta/esc50.csv
  nasa_external/
    small_uav_acoustics.zip # MATLAB recordings; external-test only
  urbansound8k/
    audio/fold1/*.flac     # optional balanced subset; through fold10
    metadata/UrbanSound8K.csv
    subset_manifest.json
  ddl/                     # ITU ARIS audio/, groups.json, splits/
```

Svanström filename prefixes map `DRONE_` to class 1 and `HELICOPTER_`/`BACKGROUND_` to class 0. ESC-50 retains `helicopter`, `airplane`, `wind`, `rain`, `thunderstorm`, `chirping_birds`, `crow`, `crickets`, `insects`, `siren`, `footsteps`, and `fireworks`. ESC excerpts are grouped by Freesound `src_file` to keep related source material in one split. UrbanSound8K is optional and retains `car_horn`, `drilling`, `engine_idling`, `gun_shot`, `jackhammer`, `siren`, and `street_music`, grouped by `fsID`.

ITU ARIS contains 4,491 drone and 1,000 background one-second clips. Adjacent clips overlap by 50%; `groups.json` defines source intervals and `splits/split_interval_disjoint.json` assigns them to train/validation/test. The pipeline must use those interval groups and preserve those fixed assignments. The alternative MLSP 2022 DDL archive is 12.6 GB and is not downloaded automatically; its `MINI`/`PRO4` codes indicate drone and `XXXX` no-drone, with segments grouped by recording-session ID. Folder-label and CSV-manifest layouts are also supported.

NASA MAT recordings contain four microphone channels and UTC timestamps. The pipeline reads each member from the ZIP, uses microphone 0 consistently, infers the 20 kHz source rate, and resamples to 16 kHz. NASA is always `external_test=True`: it is never assigned to train/validation, feature-ranked, or used for scaler fitting. `src/convert_nasa_archive.py` can create compact FLACs and conversion provenance when extra disk space is available.

## Local Setup

Python 3.10 or newer is recommended. The dependencies are CPU-capable and do not require a GPU or CUDA.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Stage 1: Curation And Features

The dataset filter copies eligible files into `data/filtered/` and writes the required `metadata.csv` inventory with file path, source dataset, original class, binary label, and scenario labels. Drone is class 1; confusers are class 0. Scenario labels are semicolon-separated when a class belongs to multiple scenarios. Scenarios are retained for later analysis and are not separate training projects.

```bash
python src/filter_datasets.py
```

For nonstandard dataset locations, override the defaults:

```bash
python src/filter_datasets.py \
  --esc50-metadata /path/to/meta/esc50.csv \
  --esc50-audio-dir /path/to/audio \
  --urbansound8k-metadata /path/to/UrbanSound8K.csv \
  --urbansound8k-audio-dir /path/to/audio
```

Extract standardized 16 kHz mono, 3-second clips as eight interpretable candidate features. Longer clips are center-cropped, shorter clips are zero-padded, and non-silent clips are peak-normalized. Final feature ranking and the choice of 4, 5, or 6 features are reserved for the leakage-safe dataset preparation stage.

```bash
python src/extract_features.py
```

To check the extraction path on a few records first:

```bash
python src/extract_features.py --limit 5 --output data/features/smoke_test.csv
```

## Stage 2: Standardization And Noise

Convert each curated recording to a mono, peak-normalized, 16 kHz, 3-second WAV. The output is separate from the curated source copies. A stable `recording_id` links every derived condition back to its original recording.

```bash
python src/preprocess_audio.py
```

Create clean copies and 20, 10, 5, and 0 dB mixtures. Confuser recordings are used as real background audio; the selected donor ID and measured SNR are saved with each row. The seed makes donor selection reproducible. This stage writes new files under `data/noisy/` and never overwrites the preprocessed audio.

```bash
python src/create_noise_sets.py
```

The same donor is used across SNR levels for a given target, making those conditions comparable. `recording_id` and `noise_recording_id` are retained in the metadata/features so the later splitting stage can keep related source material together and prevent derived-copy leakage. A usable noise set requires at least one class-0 recording distinct from each target; for robust background variety, use multiple confuser recordings.

## Project Status

## Stage 3: Ingestion, Splits, And Feature Selection

Print a preflight report and build one inventory from Svanström, selected ESC-50, NASA external audio, and any available optional UrbanSound8K/DDL data:

```bash
python src/build_master_metadata.py
```

For a nonstandard DDL layout:

```bash
python src/build_master_metadata.py --ddl-metadata /path/to/ddl_manifest.csv
```

The legacy `data/raw/drone/` directory is unused; Svanström supplies the positive class. UrbanSound8K is optional. To download only a balanced subset of selected confusers without retaining its 6 GB archive:

```bash
python src/download_urbansound_subset.py --per-class 40 --seed 42
```

Prepare the stratified 80/20 held-out split, a validation split from training recordings, segmented 16 kHz/3-second audio, candidate features, and split-local clean/20/10/5/0 dB noise variants:

```bash
python src/prepare_dataset.py
```

Splitting occurs on `recording_id` before segmentation. Every segment and noisy derivative inherits its source recording's split. Noise donors come only from the same split. Silent target/donor audio is skipped for noise generation while its clean example remains. By default noisy mixes are generated in memory and their feature rows retain target/donor IDs, requested/measured SNR, and provenance without another audio copy. Use `--store-noisy-audio` to write noisy FLACs, or `--no-noise` to omit noise conditions. The runner asserts all split intersections are empty, noisy derivatives stay in their target split, and NASA has zero rows in training, validation, feature ranking, and scaler fitting.

Outputs include `data/master_metadata.csv`, processed audio under `data/processed/{train,validation,test}/` and `data/processed/nasa_external/`, candidate feature tables under `data/features/`, `results/recording_split.csv`, `results/split_summary.csv`, `results/feature_selection.csv`, `results/selected_features.json`, balanced recording-level subset manifests in `results/small_data_subsets/`, and training-fitted `models/scaler_4.pkl`, `scaler_5.pkl`, and `scaler_6.pkl`. Standardized feature matrices are saved separately as `{split}_features_4.csv`, `_5.csv`, and `_6.csv`. Scalers are fit on training rows only, then applied to validation, test, and NASA rows.

Feature ranking aggregates segments to a recording-level mean and uses mutual information plus ANOVA F-score on training only. The combined ranking defines top 4/5/6 feature sets and their 4/5/6-qubit mappings. Small-data manifests select balanced unique training recordings and list every selected segment/condition `sample_id` to preserve source independence.

For datasets too small to support a requested subset size, each manifest records the actual balanced number of unique recordings. At least two source recordings per class are needed for a held-out split. NASA may be unlabeled; its features and training-scaler transforms are prepared for later external evaluation, but NASA never enters fitting or ranking.

Implemented: project scaffold; real Svanström, selected ESC-50, ITU ARIS, and NASA ingestion; recording/group-safe splits; training-only feature ranking and scalers; balanced small-data manifests; and classical RBF SVM/MLP baselines. Quantum-kernel training, simulator experiments, quantum noise studies, and IBM QPU execution are later stages. No IBM hardware job is submitted by any current script.

## Stage 4: Classical Baselines

Run the CPU-only RBF SVM and small MLP using the same Stage 3 recording manifests, fixed validation/test sets, selected features, and saved training-fitted scalers:

```bash
python src/train_svm.py
python src/train_mlp.py
```

Each runner uses the real balanced sizes available (24, 50, and 70 unique training recordings), all 4/5/6-feature sets, and clean/20/10/5/0 dB training conditions. SVM searches only `C={0.1,1,10}` and `gamma={scale,auto}`, selecting by internal validation F1, then balanced accuracy and drone recall. The MLP uses 16- and 8-unit ReLU layers, balanced sample weights, batch size 16, and patience-based early stopping on the same fixed recording-disjoint validation split (not an internal random row split). Neither runner fits a scaler or uses NASA for selection. A clean-trained model is also evaluated across all held-out test SNRs. Test metrics are recorded after validation selection.

Outputs: `results/svm_results.csv`, `results/mlp_results.csv`, `results/classical_comparison.csv`, `results/classical_training_sample_ids.csv`, `results/best_svm_configuration.json`, `results/best_mlp_configuration.json`, plots in `results/plots/`, and fitted estimators in `models/svm/` and `models/mlp/`.

On the current run, the strongest overall clean validation-selected configuration was the 6-feature SVM trained with 70 recordings (`C=10`, `gamma=scale`): validation F1 0.922; held-out test accuracy 0.879, F1 0.906, and balanced accuracy 0.870. The best 4-feature classical baseline for the first quantum comparison is the RBF SVM with 24 recordings (`C=0.1`, `gamma=scale`): validation F1 0.884; test accuracy 0.828, F1 0.880, ROC-AUC 0.789, and balanced accuracy 0.765. The validation-selected MLP configuration is 4 features with 70 recordings; it stopped after 29 epochs on the fixed validation split. Selection uses internal validation only; these results do not imply quantum advantage.

Grid-mean clean test accuracy for 24/50/70 training recordings is SVM 0.828/0.766/0.840 and MLP 0.819/0.815/0.822. Thus, SVM is stronger at 24 and 70, while MLP is stronger at 50. Mean clean-test accuracy by 4/5/6 features is SVM 0.803/0.793/0.838 and MLP 0.825/0.810/0.821. For clean-trained noise robustness, mean SVM F1 falls from 0.863 clean to 0.562 at 0 dB; MLP falls from 0.875 to 0.597. Mean drone recall falls from 0.917 to 0.421 for SVM and from 0.973 to 0.461 for MLP. These are grid means over the recorded configurations, not single-model claims.

The current dataset has many more background/confuser groups than drone groups. The balanced subset manifests therefore provide 24, 50, and at most 70 unique recordings, not 100 or 200. NASA recordings are unlabeled in the downloaded release, so NASA result rows and `nasa_external_comparison.png` report prediction-confidence distributions only; accuracy, recall, and F1 are intentionally blank. The selected SVM predicted 11.1% of NASA segments as drone (mean probability 0.252); the selected MLP predicted 1.0% (mean probability 0.161). These are prediction summaries, not external accuracy. NASA was not used for model selection.

## Stage 5: Ideal Quantum Simulator

Run only the smallest 4-qubit, 24-recording clean comparison first:

```bash
python src/quantum_kernel_simulator.py --initial-only
```

After the initial comparison succeeds, run the bounded fixed-kernel ideal-simulator expansion:

```bash
python src/quantum_kernel_simulator.py
```

The Stage 5 module imports no IBM Runtime, backend, credential, or hardware-job APIs. It aggregates standardized feature rows to one mean vector per recording for both the quantum and recomputed classical comparators. The initial manifest records exact constituent `sample_id`s, split intersections, and the SHA-256 of the existing Stage 3 scaler. Fixed and trainable kernel matrices are cached under `results/quantum/kernels/`; result rows are saved to `results/quantum/ideal_simulator_results.csv`.

Initial matched 4-feature/24-recording clean test results after recording-level aggregation: trainable QSVC accuracy 0.878, F1 0.684, balanced accuracy 0.851; MLP accuracy 0.878, F1 0.625, balanced accuracy 0.776; fixed QSVC accuracy 0.806, F1 0.558, balanced accuracy 0.784; recomputed RBF SVM accuracy 0.684, F1 0.492, balanced accuracy 0.786. These aggregated results are not directly comparable to the Stage 4 segment-level SVM score; the recomputed SVM above uses the same recording-level samples as QSVC.

In the fixed-kernel expansion, the best clean held-out QSVC F1 observed was 0.600 with 5 features/5 qubits and 24 recordings. The highest clean test accuracy was 0.888 with 4- or 5-feature QSVC and 70 recordings, but drone recall was only 0.500. Under noisy test conditions, drone recall was zero for most fixed-kernel configurations; the largest observed in this expansion was 0.188 at 0 dB for 4 features/24 recordings. Accuracy alone is misleading where the model predicts the majority class. The initial trainable 4-qubit result is a small-data signal to investigate, not evidence of quantum advantage.

No NASA rows enter Stage 5. No IBM job has been submitted.

## Stage 6: Controlled Noisy Quantum Simulation

Run the frozen Stage 5 trainable 4-qubit kernel through a local Aer noise sweep:

```bash
python src/quantum_noise_experiments.py --shots 1024 --seed 42
```

The runner uses the same 24 balanced training recordings, recording-disjoint 80-recording validation set, 98-recording clean test set, Stage 3 scaler, frozen Stage 5 trainable parameters, and global state-fidelity definition. It does not optimize the feature map again. Aer applies synthetic depolarizing noise to transpiled one- and two-qubit gates and symmetric readout flips; the combined sweep labels are `p2={0, 0.1%, 0.25%, 0.5%, 1%, 2%}`, with `p1=p2/10` and readout probability `p2/2`. Gate-only and readout-only ablations are also run at the 0.5% label. These are controlled simulator settings, not measured or claimed IBM backend error rates. No IBM Runtime, credentials, hardware backend, or QPU job is used.

On the current 1,024-shot run, combined-noise test F1 across 0–2% was 0.650–0.667; balanced accuracy was 0.839–0.845. F1 remained above the matched Stage 5 MLP (0.625) and RBF SVM (0.492) thresholds at the highest tested 2% label. This only establishes that no crossing occurred on the tested grid; it does not establish a hardware tolerance or quantum advantage. At zero noise, finite-shot F1 was 0.667 versus the Stage 5 ideal result of 0.684. The 0.5% gate-only and readout-only ablations each had F1 0.667. Reported drone and non-drone recalls are retained separately because accuracy and aggregate metrics can hide class imbalance.

Results and noise definitions are saved to `results/quantum/noisy_simulator_results.csv`, `results/quantum/noise_model_metadata.json`, and `results/quantum/noise_threshold_summary.json`. Finite-shot kernel matrices and fingerprints are cached under `results/quantum/noisy_kernels/`. Plots are saved under `results/plots/`. The runner also creates `models/quantum/stage7_frozen_config.json`, marked `prepared_not_submitted` with hardware submission disabled. Stage 7 remains gated: do not run hardware unless the user explicitly says `RUN REAL QPU`. NASA remains external-only and is not loaded or scored in this analysis.

## Frontend: Streamlit Dashboard (Read-Only)

Launch the website frontend (DETECT, RESEARCH, ABOUT):

```bash
streamlit run app.py
```

Behavior and guardrails:

- The frontend reads only local artifacts under `results/` and `models/`.
- No IBM Runtime submission, polling, cancellation, or QPU-control path is called from the app.
- DETECT performs local inference only, using existing saved model/scaler artifacts.
- The HOME hero image is loaded from the project-root `background.jpg`.
- Optional ambience can be added as `assets/ambience.mp3` and is off by default.
# QubitSky Datasets

QubitSky uses a small set of audio datasets for drone detection.

## 1. DDL — Main Drone Dataset

Source:
https://zenodo.org/records/6459183

Use:
- Drone audio = label 1
- No-drone audio = label 0
- Main dataset for training and testing

Folder:
data/raw/ddl/

Prefer the real field recordings.

---

## 2. ESC-50 — Environmental Noise

Source:
https://github.com/karolpiczak/ESC-50

Use only useful drone-confuser sounds such as:

- helicopter
- airplane
- wind
- rain
- thunderstorm
- birds
- crow
- crickets
- insects
- siren
- footsteps
- fireworks

Do NOT use unrelated classes like:
- crying baby
- toilet flush
- brushing teeth
- clock tick

Main purpose:
Use these sounds as realistic background noise and confusers.

Folder:
data/raw/esc50/

---

## 3. UrbanSound8K — Urban Noise

Source:
https://urbansounddataset.weebly.com/

Useful classes:

- engine_idling
- siren
- car_horn
- gun_shot
- street_music
- drilling
- jackhammer

Main purpose:
Urban and mechanical background noise.

Folder:
data/raw/urbansound8k/

---

## 4. NASA Small UAS — Final External Test

Source:
https://data.nasa.gov/dataset/small-uas-flyover-acoustics-data

Use:
- Final testing only
- Do NOT train on NASA data

Purpose:
Test whether the trained model can recognize drone recordings from a completely different source.

Folder:
data/raw/nasa_external/

---

# Labels

1 = Drone

0 = Non-drone / background sound

---

# Noise Levels

Create noisy test audio at:

- Clean
- 20 dB SNR
- 10 dB SNR
- 5 dB SNR
- 0 dB SNR

Use ESC-50 and UrbanSound8K sounds as background noise.

---

# Dataset Roles

| Dataset | Purpose |
|---|---|
| DDL | Main training/testing |
| ESC-50 | Environmental noise/confusers |
| UrbanSound8K | Urban/mechanical noise |
| NASA Small UAS | Final unseen test only |

---

# Important Rule

If one recording is split into multiple 3-second clips, keep all clips from that recording in the same train/test split.

Also keep all noisy versions of the same recording in the same split.

This prevents data leakage.

---

# Final Flow

DDL
→ Train/Test

ESC-50 + UrbanSound8K
→ Add realistic noise

Audio
→ 16 kHz
→ 3-second clips
→ Extract 4–6 features
→ SVM vs MLP vs Quantum Kernel

NASA
→ Final unseen external test
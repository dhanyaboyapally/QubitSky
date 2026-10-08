"""Shared paths and experiment settings for QubitSky."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FILTERED_DIR = DATA_DIR / "filtered"
PREPROCESSED_DIR = FILTERED_DIR / "preprocessed"
PREPROCESSED_METADATA = FILTERED_DIR / "preprocessed_metadata.csv"
PROCESSED_DIR = DATA_DIR / "processed"
NOISY_DIR = DATA_DIR / "noisy"
NOISY_AUDIO_DIR = NOISY_DIR / "audio"
NOISY_METADATA = NOISY_DIR / "metadata.csv"
FEATURES_DIR = DATA_DIR / "features"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
DDL_ROOT = RAW_DIR / "ddl"
ITU_ARIS_ROOT = DDL_ROOT
ITU_GROUPS_METADATA = ITU_ARIS_ROOT / "groups.json"
ITU_DISJOINT_SPLIT = ITU_ARIS_ROOT / "splits" / "split_interval_disjoint.json"
SVANSTROM_ROOT = RAW_DIR / "svanstrom"
NASA_EXTERNAL_ROOT = RAW_DIR / "nasa_external"
NASA_ARCHIVE = NASA_EXTERNAL_ROOT / "small_uav_acoustics.zip"
NASA_AUDIO_DIR = NASA_EXTERNAL_ROOT / "audio"
NASA_CONVERSION_METADATA = NASA_EXTERNAL_ROOT / "conversion_metadata.csv"
MASTER_METADATA = DATA_DIR / "master_metadata.csv"
SPLIT_SUMMARY = RESULTS_DIR / "split_summary.csv"
FEATURE_SELECTION_CSV = RESULTS_DIR / "feature_selection.csv"
SELECTED_FEATURES_JSON = RESULTS_DIR / "selected_features.json"

DRONE_AUDIO_DIR = RAW_DIR / "drone"
ESC50_ROOT = RAW_DIR / "esc50"
ESC50_METADATA = ESC50_ROOT / "meta" / "esc50.csv"
ESC50_AUDIO_DIR = ESC50_ROOT / "audio"
URBANSOUND8K_ROOT = RAW_DIR / "urbansound8k"
URBANSOUND8K_METADATA = URBANSOUND8K_ROOT / "metadata" / "UrbanSound8K.csv"
URBANSOUND8K_AUDIO_DIR = URBANSOUND8K_ROOT / "audio"

METADATA_FILENAME = "metadata.csv"
FEATURES_FILENAME = "features.csv"
SAMPLE_RATE = 16_000
# One-second windows match the native ITU-ARIS clip length, so no source needs
# zero-padding. Padding 1 s clips to 3 s made "is this ITU-ARIS?" recoverable from
# the features and let models learn the dataset instead of the drone.
CLIP_DURATION_SECONDS = 1.0
CLIP_SAMPLE_COUNT = int(SAMPLE_RATE * CLIP_DURATION_SECONDS)
RANDOM_SEED = 42
SNR_LEVELS_DB = (20, 10, 5, 0)

ESC50_CONFUSERS = {
    "airplane",
    "chirping_birds",
    "crickets",
    "crow",
    "fireworks",
    "helicopter",
    "insects",
    "rain",
    "siren",
    "thunderstorm",
    "wind",
    "footsteps",
}

REQUIRED_DATASETS = {"svanstrom", "esc50", "nasa_external"}
OPTIONAL_DATASETS = {"urbansound8k", "ddl"}

URBANSOUND8K_CONFUSERS = {
    "car_horn",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
}

SCENARIO_CLASSES = {
    "wildlife": {"birds", "chirping_birds", "crow", "crickets", "insects", "wind"},
    "infrastructure": {"wind", "rain", "engine_idling", "engines", "insects"},
    "battlefield/noisy-field": {"gun_shot", "engines", "engine_idling", "helicopter", "airplane"},
    "perimeter": {"engine_idling", "engines", "siren", "footsteps", "wind"},
    "event": {"clapping", "laughing", "street_music", "music"},
    "airspace": {"helicopter", "airplane"},
}

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aiff", ".aif"}
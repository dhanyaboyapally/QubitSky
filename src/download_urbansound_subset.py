"""Stream a balanced UrbanSound8K confuser subset into the local raw-data tree."""

import argparse
import csv
import io
import json
import random
import re
import ssl
import tarfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf
import certifi

from config import PROJECT_ROOT, URBANSOUND8K_CONFUSERS, URBANSOUND8K_ROOT


ARCHIVE_URL = "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz?download=1"
CLASS_IDS = {
    1: "car_horn",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music",
}
MEMBER_PATTERN = re.compile(r"(?P<fsid>\d+)-(?P<class_id>\d+)-\d+-\d+\.wav$", re.I)


def stream_subset(
    output_root: Path,
    per_class: int = 40,
    seed: int = 42,
) -> dict[str, int]:
    if per_class < 1:
        raise ValueError("per_class must be positive")
    expected_classes = set(URBANSOUND8K_CONFUSERS)
    if expected_classes != set(CLASS_IDS.values()):
        raise ValueError("UrbanSound8K class IDs and configured confuser classes disagree")

    rng = random.Random(seed)
    selected: dict[int, list[tuple[str, bytes]]] = {class_id: [] for class_id in CLASS_IDS}
    seen: dict[int, int] = {class_id: 0 for class_id in CLASS_IDS}
    metadata_content: bytes | None = None
    request = Request(ARCHIVE_URL, headers={"User-Agent": "QubitSky dataset preparation"})

    tls_context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, timeout=120, context=tls_context) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                normalized_member = member.name.replace("\\", "/")
                if normalized_member.endswith("/metadata/UrbanSound8K.csv"):
                    source = archive.extractfile(member)
                    if source is not None:
                        metadata_content = source.read()
                    continue

                if "/audio/fold" not in normalized_member or not normalized_member.lower().endswith(".wav"):
                    continue
                match = MEMBER_PATTERN.search(Path(normalized_member).name)
                if match is None:
                    continue
                class_id = int(match.group("class_id"))
                if class_id not in selected:
                    continue
                source = archive.extractfile(member)
                if source is None:
                    continue
                payload = source.read()
                seen[class_id] += 1
                reservoir = selected[class_id]
                if len(reservoir) < per_class:
                    reservoir.append((normalized_member, payload))
                else:
                    replacement = rng.randrange(seen[class_id])
                    if replacement < per_class:
                        reservoir[replacement] = (normalized_member, payload)

    if metadata_content is None:
        raise ValueError("UrbanSound8K archive did not contain its metadata CSV")
    missing_classes = [CLASS_IDS[class_id] for class_id, rows in selected.items() if not rows]
    if missing_classes:
        raise ValueError(f"UrbanSound8K archive contained no selected clips for: {missing_classes}")

    (output_root / "metadata").mkdir(parents=True, exist_ok=True)
    (output_root / "audio").mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "metadata" / "UrbanSound8K.csv"
    metadata_path.write_bytes(metadata_content)

    saved_counts: dict[str, int] = {}
    for class_id, class_name in CLASS_IDS.items():
        saved = 0
        for member_name, payload in selected[class_id]:
            fold_match = re.search(r"/audio/(fold\d+)/", member_name)
            if fold_match is None:
                raise ValueError(f"Could not infer UrbanSound8K fold from {member_name}")
            fold = fold_match.group(1)
            relative_path = Path(member_name)
            filename = relative_path.name
            audio_root = output_root / "audio" / fold
            audio_root.mkdir(parents=True, exist_ok=True)
            with sf.SoundFile(io.BytesIO(payload)) as source_audio:
                samples = source_audio.read(dtype="float32", always_2d=True)
                sample_rate = source_audio.samplerate
            destination = audio_root / f"{Path(filename).stem}.flac"
            sf.write(
                destination,
                samples,
                sample_rate,
                format="FLAC",
                subtype="PCM_24",
            )
            saved += 1
        saved_counts[class_name] = saved

    manifest = {
        "source": ARCHIVE_URL,
        "selection": "uniform reservoir sample within each requested class",
        "seed": seed,
        "per_class_target": per_class,
        "archive_class_files_seen": {
            CLASS_IDS[class_id]: count for class_id, count in seen.items()
        },
        "stored_flac_files": saved_counts,
        "metadata_csv": metadata_path.relative_to(PROJECT_ROOT).as_posix(),
        "audio_format": "lossless FLAC; source sample rate and channels preserved",
    }
    manifest_path = output_root / "subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return saved_counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=URBANSOUND8K_ROOT)
    parser.add_argument("--per-class", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    counts = stream_subset(args.output_root, args.per_class, args.seed)
    print(f"Saved balanced UrbanSound8K confuser subset: {counts}")
    print(f"Total clips saved: {sum(counts.values())}")


if __name__ == "__main__":
    main()
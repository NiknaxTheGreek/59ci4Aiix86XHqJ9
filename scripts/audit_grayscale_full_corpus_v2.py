from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

from src.grayscale_v2 import rec709_grayscale

EXPECTED_ARCHIVE_BYTES = 939_921_132
EXPECTED_ARCHIVE_SHA256 = "033dd76fa617ba9bcf16d8ca4dcc294a838a6956eae0740f3013e4663805008f"
EXPECTED_MANIFEST_SHA256 = "3fb8a29f1a51fc7930fe6c876d0fcb8d99ec59e147d4abbff503da4870f15b0b"
EXPECTED_O_SPLIT_SHA256 = "d48e182da1b170365b39693317e1edf6dd3395c02ccabdd6a771d7be5e226680"
EXPECTED_N = 2989


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pixel_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def decode_member(z: ZipFile, member: str) -> np.ndarray:
    with Image.open(io.BytesIO(z.read(member))) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def make_thumb(array: np.ndarray, width: int = 180, height: int = 320) -> Image.Image:
    return Image.fromarray(array, mode="RGB").resize(
        (width, height), Image.Resampling.BILINEAR
    )


def choose_qa_rows(meta_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in meta_rows:
        groups[(row["environment_id"], row["label"], row["supplied_split"])].append(row)

    expected_groups = {
        (env, label, supplied)
        for env in ("ENV-01", "ENV-02", "ENV-03", "ENV-04")
        for label in ("flip", "notflip")
        for supplied in ("training", "testing")
    }
    if set(groups) != expected_groups:
        missing = sorted(expected_groups - set(groups))
        extra = sorted(set(groups) - expected_groups)
        raise RuntimeError(f"Unexpected QA strata. missing={missing} extra={extra}")

    selected: list[dict[str, str]] = []
    for key in sorted(groups):
        rows = sorted(
            groups[key],
            key=lambda r: (r["video_id"], int(r["frame_number"]), r["sample_id"]),
        )
        selected.append(rows[0])
        selected.append(rows[-1])
    if len(selected) != 32 or len({r["sample_id"] for r in selected}) != 32:
        raise RuntimeError("Expected exactly 32 unique deterministic QA images")
    return selected


def write_contact_sheets(
    qa_rows: list[dict[str, str]],
    qa_images: dict[str, tuple[Image.Image, Image.Image]],
    output_dir: Path,
) -> list[str]:
    files: list[str] = []
    per_sheet = 8
    for sheet_index in range(0, len(qa_rows), per_sheet):
        batch = qa_rows[sheet_index:sheet_index + per_sheet]
        width = 410
        row_h = 370
        header_h = 34
        canvas = Image.new("RGB", (width, header_h + row_h * len(batch)), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 8), "ORIGINAL", fill="black")
        draw.text((220, 8), "GRAYSCALE (Rec.709)", fill="black")

        for i, row in enumerate(batch):
            y = header_h + i * row_h
            original, gray = qa_images[row["sample_id"]]
            canvas.paste(original, (20, y))
            canvas.paste(gray, (220, y))
            caption = (
                f'{row["sample_id"]} | {row["environment_id"]} | '
                f'{row["label"]} | {row["supplied_split"]}'
            )
            draw.text((20, y + 325), caption, fill="black")

        name = f"grayscale_qa_{sheet_index // per_sheet + 1:02d}.jpg"
        canvas.save(output_dir / name, quality=94, subsampling=0)
        files.append(name)
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    if args.archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError("Authoritative archive byte size changed")
    if sha256_file(args.archive) != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("Authoritative archive SHA-256 changed")
    if sha256_file(args.manifest) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("DATA-001 manifest SHA-256 changed")
    if sha256_file(args.metadata) != EXPECTED_O_SPLIT_SHA256:
        raise RuntimeError("Frozen O membership SHA-256 changed")

    manifest = read_csv(args.manifest)
    metadata = read_csv(args.metadata)
    if len(manifest) != EXPECTED_N or len(metadata) != EXPECTED_N:
        raise RuntimeError("Expected exactly 2,989 manifest and metadata rows")
    if len({r["sample_id"] for r in manifest}) != EXPECTED_N:
        raise RuntimeError("Dataset manifest sample IDs are not unique")
    if len({r["sample_id"] for r in metadata}) != EXPECTED_N:
        raise RuntimeError("Metadata sample IDs are not unique")

    meta_by_id = {r["sample_id"]: r for r in metadata}
    if set(meta_by_id) != {r["sample_id"] for r in manifest}:
        raise RuntimeError("Manifest/metadata sample-ID populations differ")

    qa_rows = choose_qa_rows(metadata)
    qa_ids = {r["sample_id"] for r in qa_rows}
    qa_images: dict[str, tuple[Image.Image, Image.Image]] = {}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, object]] = []
    aggregate = hashlib.sha256()
    failures: list[str] = []
    channel_identity_passes = 0
    shape_passes = 0

    with ZipFile(args.archive) as z:
        names = set(z.namelist())
        for i, source in enumerate(manifest, 1):
            sid = source["sample_id"]
            member = source["archive_member"]
            if member not in names:
                failures.append(f"{sid}:missing_archive_member")
                continue
            try:
                rgb = decode_member(z, member)
                gray = rec709_grayscale(rgb)
            except Exception as exc:
                failures.append(f"{sid}:{type(exc).__name__}:{exc}")
                continue

            if gray.shape == rgb.shape:
                shape_passes += 1
            else:
                failures.append(f"{sid}:shape_changed:{rgb.shape}->{gray.shape}")
                continue

            channels_identical = bool(
                np.array_equal(gray[..., 0], gray[..., 1])
                and np.array_equal(gray[..., 1], gray[..., 2])
            )
            if channels_identical:
                channel_identity_passes += 1
            else:
                failures.append(f"{sid}:grayscale_channels_not_identical")
                continue

            gray_hash = pixel_sha256(gray)
            rgb_hash = pixel_sha256(rgb)
            aggregate.update(sid.encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(gray_hash.encode("ascii"))
            aggregate.update(b"\n")

            meta = meta_by_id[sid]
            audit_rows.append({
                "sample_id": sid,
                "label": meta["label"],
                "video_id": meta["video_id"],
                "frame_number": int(meta["frame_number"]),
                "environment_id": meta["environment_id"],
                "supplied_split": meta["supplied_split"],
                "archive_member": member,
                "height": int(rgb.shape[0]),
                "width": int(rgb.shape[1]),
                "original_rgb_pixel_sha256": rgb_hash,
                "grayscale_rgb_pixel_sha256": gray_hash,
                "channels_identical": channels_identical,
            })

            if sid in qa_ids:
                qa_images[sid] = (make_thumb(rgb), make_thumb(gray))

            if i % 250 == 0 or i == EXPECTED_N:
                print(f"grayscale_audit_progress={i}/{EXPECTED_N}", flush=True)

    if failures:
        (args.output_dir / "failures.json").write_text(
            json.dumps(failures, indent=2) + "\n", encoding="utf-8"
        )
        raise RuntimeError(f"Full-corpus grayscale audit had {len(failures)} failures")

    if len(audit_rows) != EXPECTED_N:
        raise RuntimeError(f"Transformed {len(audit_rows)} images, expected 2,989")
    if shape_passes != EXPECTED_N or channel_identity_passes != EXPECTED_N:
        raise RuntimeError("Full-corpus transform invariants did not pass 2,989/2,989")
    if set(qa_images) != qa_ids:
        raise RuntimeError("Not all deterministic QA examples were captured")

    audit_path = args.output_dir / "GRAYSCALE_FULL_CORPUS_MANIFEST.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    qa_csv = args.output_dir / "GRAYSCALE_QA_SAMPLE_32.csv"
    with qa_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(qa_rows[0]))
        writer.writeheader()
        writer.writerows(qa_rows)

    contact_sheets = write_contact_sheets(qa_rows, qa_images, args.output_dir)

    counts = {
        "labels": dict(Counter(r["label"] for r in metadata)),
        "environments": dict(Counter(r["environment_id"] for r in metadata)),
        "supplied_splits": dict(Counter(r["supplied_split"] for r in metadata)),
    }
    summary = {
        "status": "PASS_FULL_CORPUS_GRAYSCALE_AUDIT",
        "transform": "Rec.709 Y=0.2126R+0.7152G+0.0722B, rounded/clipped uint8, repeated to RGB",
        "source_resolution_transform": True,
        "images_expected": EXPECTED_N,
        "images_transformed": len(audit_rows),
        "decode_transform_failures": 0,
        "shape_preserved": shape_passes,
        "three_output_channels_identical": channel_identity_passes,
        "unique_sample_ids": len({r["sample_id"] for r in audit_rows}),
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "dataset_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "metadata_split_sha256": EXPECTED_O_SPLIT_SHA256,
        "grayscale_audit_manifest_sha256": sha256_file(audit_path),
        "aggregate_grayscale_identity_sha256": aggregate.hexdigest(),
        "qa_sample_n": len(qa_rows),
        "qa_contact_sheets": contact_sheets,
        "qa_stratification": "2 deterministic samples per environment x label x supplied_split stratum",
        "visual_gate": "ACCEPTED_BY_USER_2026-10-04_BEFORE_FULL_RUN",
        "counts": counts,
    }
    (args.output_dir / "GRAYSCALE_FULL_CORPUS_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

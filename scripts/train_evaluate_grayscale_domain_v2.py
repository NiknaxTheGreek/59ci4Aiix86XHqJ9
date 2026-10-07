from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from src.chroma_counterfactual_v2 import rec709_grayscale
from src.scratch_v2 import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    EXPECTED_ARCHIVE_BYTES,
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_BANK_SHA256,
    EXPECTED_DATASET_MANIFEST_SHA256,
    EXPECTED_SPLIT_SHA256,
    MemmapDataset,
    ScratchCNN,
    ScratchCNNConfig,
    build_optimizer,
    candidate_config,
    count_trainable_parameters,
    load_candidate_bank,
    make_generator,
    preprocess_to_uint8,
    read_csv,
    resolved_optimizer_settings,
    seed_everything,
    sha256_file,
)

CHAMPIONS = {"O": "C14", "S": "C18", "T": "C06", "ST": "C07"}
EXPECTED_COUNTS = {
    ("O", "train"): 2392, ("O", "test"): 597,
    ("S", "train"): 2219, ("S", "test"): 770,
    ("T", "train"): 2365, ("T", "test"): 624,
    ("ST", "train"): 1756, ("ST", "test"): 161,
}
EXPECTED_ORIGINAL_F1 = {
    "O": 0.998272884283247,
    "S": 0.8196248196248196,
    "T": 0.88,
    "ST": 0.704,
}
EXPECTED_TESTTIME_GRAY_F1 = {
    "O": 0.6839622641509434,
    "S": 0.6225402504472272,
    "T": 0.6695095948827292,
    "ST": 0.7109004739336493,
}
N_BOOT = 5000
BOOT_SEED = 20261004


def decode_member(z: ZipFile, member: str) -> np.ndarray:
    with Image.open(io.BytesIO(z.read(member))) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def pixel_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metrics(y_true: np.ndarray, probability: np.ndarray) -> dict:
    pred = (probability >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y_true)),
    }


def ci(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def bootstrap_delta(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> dict:
    frame = np.empty(N_BOOT, dtype=np.float64)
    group = np.empty(N_BOOT, dtype=np.float64)

    rng = np.random.default_rng(seed)
    for i in range(N_BOOT):
        idx = rng.integers(0, len(y), size=len(y))
        frame[i] = (
            f1_score(y[idx], pred_b[idx], zero_division=0)
            - f1_score(y[idx], pred_a[idx], zero_division=0)
        )

    unique = np.asarray(sorted(set(groups.tolist())), dtype=object)
    by_group = {g: np.where(groups == g)[0] for g in unique}
    rng = np.random.default_rng(seed + 1)
    for i in range(N_BOOT):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_group[g] for g in sampled])
        group[i] = (
            f1_score(y[idx], pred_b[idx], zero_division=0)
            - f1_score(y[idx], pred_a[idx], zero_division=0)
        )

    return {
        "observed_delta_f1": float(
            f1_score(y, pred_b, zero_division=0) - f1_score(y, pred_a, zero_division=0)
        ),
        "frame_delta_f1_ci95": ci(frame),
        "video_group_delta_f1_ci95": ci(group),
        "resamples": N_BOOT,
    }


def load_audit(path: Path) -> tuple[dict[str, dict[str, str]], str]:
    rows = read_csv(path)
    if len(rows) != 2989:
        raise RuntimeError(f"Audit manifest must contain 2,989 rows, found {len(rows)}")
    by_id = {r["sample_id"]: r for r in rows}
    if len(by_id) != 2989:
        raise RuntimeError("Audit manifest sample IDs are not unique")
    if any(r.get("channels_identical") not in ("True", "true", "1") for r in rows):
        raise RuntimeError("Audit manifest contains a non-grayscale row")
    return by_id, sha256_file(path)


def build_grayscale_cache(
    *,
    archive_path: Path,
    manifest_by_id: dict[str, dict[str, str]],
    membership: list[dict[str, str]],
    audit_by_id: dict[str, dict[str, str]],
    regime: str,
    role: str,
    output_root: Path,
) -> tuple[Path, Path, dict]:
    selected = [r for r in membership if r["role"] == role]
    expected_n = EXPECTED_COUNTS[(regime, role)]
    if len(selected) != expected_n:
        raise RuntimeError(f"{regime}/{role} count {len(selected)} != {expected_n}")

    data_path = output_root / f"{regime}_{role}_gray.npy"
    index_path = output_root / f"{regime}_{role}_gray_index.csv"
    cache = np.lib.format.open_memmap(
        data_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(selected), 3, CANVAS_HEIGHT, CANVAS_WIDTH),
    )
    index_rows: list[dict] = []

    with ZipFile(archive_path) as z:
        for i, member in enumerate(selected, 1):
            sid = member["sample_id"]
            source = manifest_by_id[sid]
            rgb = decode_member(z, source["archive_member"])
            gray = rec709_grayscale(rgb)
            if pixel_sha256(gray) != audit_by_id[sid]["grayscale_rgb_pixel_sha256"]:
                raise RuntimeError(f"Grayscale identity mismatch against full audit: {sid}")
            cache[i - 1] = preprocess_to_uint8(Image.fromarray(gray, mode="RGB"))
            index_rows.append({
                "cache_index": i - 1,
                "sample_id": sid,
                "label": member["label"],
                "video_id": member["video_id"],
                "frame_number": int(member["frame_number"]),
                "environment_id": member["environment_id"],
                "supplied_split": member["supplied_split"],
                "regime": regime,
                "role": role,
                "archive_member": source["archive_member"],
            })
            if i % 250 == 0 or i == len(selected):
                print(f"gray_cache_progress regime={regime} role={role} {i}/{len(selected)}", flush=True)

    cache.flush()
    del cache
    write_csv(index_path, index_rows, list(index_rows[0]))
    receipt = {
        "regime": regime,
        "role": role,
        "rows": len(index_rows),
        "shape": [len(index_rows), 3, CANVAS_HEIGHT, CANVAS_WIDTH],
        "dtype": "uint8",
        "transform": "source-resolution Rec.709 grayscale then frozen 398x224 preprocessing",
        "cache_data_sha256": sha256_file(data_path),
        "cache_index_sha256": sha256_file(index_path),
    }
    return data_path, index_path, receipt


def load_external_predictions(
    path: Path,
    test_rows: list[dict[str, str]],
    *,
    candidate_id: str | None,
    probability_field: str,
) -> np.ndarray:
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"Prediction file is empty: {path}")
    if candidate_id is not None and "candidate_id" in rows[0]:
        rows = [r for r in rows if r["candidate_id"] == candidate_id]
    by_id = {r["sample_id"]: r for r in rows}
    if len(by_id) != len(test_rows):
        raise RuntimeError(
            f"Prediction population mismatch for {path}: {len(by_id)} != {len(test_rows)}"
        )
    if set(by_id) != {r["sample_id"] for r in test_rows}:
        raise RuntimeError(f"Prediction sample IDs differ from frozen test set: {path}")
    return np.asarray([float(by_id[r["sample_id"]][probability_field]) for r in test_rows])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True, choices=("O", "S", "T", "ST"))
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--membership", required=True, type=Path)
    ap.add_argument("--bank", required=True, type=Path)
    ap.add_argument("--audit-manifest", required=True, type=Path)
    ap.add_argument("--testtime-gray-predictions", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    regime = args.regime
    candidate_id = args.candidate_id
    if CHAMPIONS[regime] != candidate_id:
        raise RuntimeError(f"{regime} must use frozen champion config {CHAMPIONS[regime]}")

    if args.archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError("Archive byte-size mismatch")
    if sha256_file(args.archive) != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("Archive SHA-256 mismatch")
    if sha256_file(args.manifest) != EXPECTED_DATASET_MANIFEST_SHA256:
        raise RuntimeError("Dataset manifest SHA-256 mismatch")
    if sha256_file(args.membership) != EXPECTED_SPLIT_SHA256[regime]:
        raise RuntimeError(f"{regime} split SHA-256 mismatch")
    if sha256_file(args.bank) != EXPECTED_BANK_SHA256:
        raise RuntimeError("Candidate bank SHA-256 mismatch")

    bank = load_candidate_bank(args.bank)
    config = candidate_config(bank, candidate_id)
    manifest = read_csv(args.manifest)
    membership = read_csv(args.membership)
    if len(manifest) != 2989 or len(membership) != 2989:
        raise RuntimeError("Manifest and regime membership must each account for 2,989 images")
    manifest_by_id = {r["sample_id"]: r for r in manifest}
    if len(manifest_by_id) != 2989:
        raise RuntimeError("Dataset manifest sample IDs are not unique")

    audit_by_id, audit_manifest_sha = load_audit(args.audit_manifest)
    if set(audit_by_id) != set(manifest_by_id):
        raise RuntimeError("Full grayscale audit population differs from DATA-001")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    work = args.output_dir / "_cache"
    work.mkdir(parents=True, exist_ok=True)

    # Build only the frozen training population before training.
    train_data, train_index, train_receipt = build_grayscale_cache(
        archive_path=args.archive,
        manifest_by_id=manifest_by_id,
        membership=membership,
        audit_by_id=audit_by_id,
        regime=regime,
        role="train",
        output_root=work,
    )

    seed = int(config["seed"])
    seed_everything(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    train_dataset = MemmapDataset(train_data, train_index)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=make_generator(seed),
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    model = ScratchCNN(ScratchCNNConfig(
        depth=int(config["depth"]),
        start_filters=int(config["start_filters"]),
        dropout=float(config["dropout"]),
    ))
    parameter_count = count_trainable_parameters(model)
    optimizer = build_optimizer(model, config)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cpu")
    model.to(device)

    history: list[dict] = []
    started = time.perf_counter()
    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0
        total_n = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            n = labels.shape[0]
            total_loss += float(loss.detach()) * n
            total_n += n
        mean_loss = total_loss / total_n
        history.append({"epoch": epoch, "train_loss": mean_loss})
        print(
            f"gray_domain regime={regime} candidate={candidate_id} "
            f"epoch={epoch:02d}/20 train_loss={mean_loss:.8f}",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    checkpoint_path = args.output_dir / f"{regime}_{candidate_id}_grayscale.pt"
    payload = {
        "protocol_version": "v2-phase11d-grayscale-domain",
        "selection_bearing": False,
        "regime": regime,
        "candidate_id": candidate_id,
        "epoch": 20,
        "candidate_config": config,
        "fixed_training": bank["fixed_training"],
        "optimizer_resolved": resolved_optimizer_settings(config),
        "parameter_count": parameter_count,
        "bank_sha256": EXPECTED_BANK_SHA256,
        "split_sha256": EXPECTED_SPLIT_SHA256[regime],
        "grayscale_audit_manifest_sha256": audit_manifest_sha,
        "train_cache_data_sha256": train_receipt["cache_data_sha256"],
        "train_cache_index_sha256": train_receipt["cache_index_sha256"],
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)

    # Mandatory round-trip freeze before primary test cache is built.
    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    fresh = ScratchCNN(ScratchCNNConfig(
        depth=int(config["depth"]),
        start_filters=int(config["start_filters"]),
        dropout=float(config["dropout"]),
    ))
    fresh.load_state_dict(reloaded["model_state_dict"], strict=True)
    fresh.eval()
    if int(reloaded["epoch"]) != 20 or reloaded["candidate_id"] != candidate_id:
        raise RuntimeError("Frozen grayscale checkpoint round-trip failed")

    # Only now build/open the frozen regime test population.
    test_data, test_index, test_receipt = build_grayscale_cache(
        archive_path=args.archive,
        manifest_by_id=manifest_by_id,
        membership=membership,
        audit_by_id=audit_by_id,
        regime=regime,
        role="test",
        output_root=work,
    )
    test_dataset = MemmapDataset(test_data, test_index)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_rows = test_dataset.rows

    y_true_list: list[int] = []
    p_graydomain_list: list[float] = []
    with torch.no_grad():
        for batch in test_loader:
            probs = torch.sigmoid(fresh(batch["image"])).cpu().numpy()
            p_graydomain_list.extend(probs.tolist())
            y_true_list.extend(batch["label"].cpu().numpy().astype(int).tolist())

    y = np.asarray(y_true_list, dtype=np.int64)
    p_graydomain = np.asarray(p_graydomain_list, dtype=np.float64)
    if len(y) != EXPECTED_COUNTS[(regime, "test")]:
        raise RuntimeError("Grayscale-domain test count changed")

    # Phase 11C recomputed original-RGB probabilities on the identical frozen
    # champion/test rows and verified them against the saved primary predictions.
    # Use that same one-row-per-test-sample evidence table here so all regimes,
    # including S, have one uniform comparison source.
    p_original = load_external_predictions(
        args.testtime_gray_predictions,
        test_rows,
        candidate_id=None,
        probability_field="p_original_rgb",
    )
    p_testtime_gray = load_external_predictions(
        args.testtime_gray_predictions,
        test_rows,
        candidate_id=None,
        probability_field="p_rec709_grayscale",
    )

    m_original = metrics(y, p_original)
    m_testtime = metrics(y, p_testtime_gray)
    m_graydomain = metrics(y, p_graydomain)
    if abs(m_original["f1"] - EXPECTED_ORIGINAL_F1[regime]) > 1e-12:
        raise RuntimeError(
            f"Original F1 drift for {regime}: {m_original['f1']} "
            f"!= {EXPECTED_ORIGINAL_F1[regime]}"
        )
    if abs(m_testtime["f1"] - EXPECTED_TESTTIME_GRAY_F1[regime]) > 1e-12:
        raise RuntimeError(
            f"Phase11C grayscale F1 drift for {regime}: {m_testtime['f1']} "
            f"!= {EXPECTED_TESTTIME_GRAY_F1[regime]}"
        )

    pred_original = (p_original >= 0.5).astype(np.int64)
    pred_testtime = (p_testtime_gray >= 0.5).astype(np.int64)
    pred_graydomain = (p_graydomain >= 0.5).astype(np.int64)
    groups = np.asarray([r["video_id"] for r in test_rows], dtype=object)
    offset = {"O": 0, "S": 100, "T": 200, "ST": 300}[regime]
    boot = {
        "grayscale_domain_minus_original_rgb": bootstrap_delta(
            y, pred_original, pred_graydomain, groups, BOOT_SEED + offset
        ),
        "testtime_grayscale_minus_original_rgb": bootstrap_delta(
            y, pred_original, pred_testtime, groups, BOOT_SEED + offset + 10
        ),
        "grayscale_domain_minus_testtime_grayscale": bootstrap_delta(
            y, pred_testtime, pred_graydomain, groups, BOOT_SEED + offset + 20
        ),
    }

    prediction_rows = []
    for row, yt, po, pt, pg in zip(
        test_rows, y, p_original, p_testtime_gray, p_graydomain, strict=True
    ):
        prediction_rows.append({
            "sample_id": row["sample_id"],
            "video_id": row["video_id"],
            "frame_number": int(row["frame_number"]),
            "y_true": int(yt),
            "p_original_rgb": float(po),
            "p_rgbtrained_grayscaletested": float(pt),
            "p_grayscaletrained_grayscaletested": float(pg),
            "pred_original_rgb": int(po >= 0.5),
            "pred_rgbtrained_grayscaletested": int(pt >= 0.5),
            "pred_grayscaletrained_grayscaletested": int(pg >= 0.5),
        })
    write_csv(
        args.output_dir / f"{regime}_predictions.csv",
        prediction_rows,
        list(prediction_rows[0]),
    )

    training_metadata = {
        "status": "FROZEN_GRAYSCALE_DOMAIN_EPOCH20",
        "selection_bearing": False,
        "regime": regime,
        "candidate_id": candidate_id,
        "candidate_config": config,
        "optimizer_resolved": resolved_optimizer_settings(config),
        "parameter_count": parameter_count,
        "train_rows": len(train_dataset),
        "test_rows": len(test_dataset),
        "validation_rows_loaded": 0,
        "epochs_completed": 20,
        "final_train_loss": history[-1]["train_loss"],
        "history": history,
        "checkpoint_sha256": checkpoint_sha,
        "bank_sha256": EXPECTED_BANK_SHA256,
        "split_sha256": EXPECTED_SPLIT_SHA256[regime],
        "grayscale_audit_manifest_sha256": audit_manifest_sha,
        "train_cache": train_receipt,
        "test_cache": test_receipt,
        "elapsed_seconds": elapsed,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    (args.output_dir / f"{regime}_training_metadata.json").write_text(
        json.dumps(training_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "status": "GRAYSCALE_DOMAIN_FIXED_CONFIG_COMPLETE",
        "selection_bearing": False,
        "retraining_performed": True,
        "regime": regime,
        "candidate_id": candidate_id,
        "threshold": 0.5,
        "epochs": 20,
        "validation": False,
        "early_stopping": False,
        "candidate_search": False,
        "transform_domain": "source-resolution Rec.709 grayscale before frozen preprocessing",
        "full_corpus_grayscale_audit_rows": 2989,
        "grayscale_audit_manifest_sha256": audit_manifest_sha,
        "test_images": len(y),
        "test_videos": len(set(groups.tolist())),
        "metrics": {
            "original_rgb_trained_rgb_tested": m_original,
            "rgb_trained_grayscale_tested": m_testtime,
            "grayscale_trained_grayscale_tested": m_graydomain,
        },
        "bootstrap": boot,
        "checkpoint_sha256": checkpoint_sha,
    }
    (args.output_dir / f"{regime}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"# Phase 11D — {regime} grayscale-domain fixed-config ablation",
        "",
        f"Frozen configuration: **{candidate_id}**. No HPO, validation, early stopping, or threshold tuning.",
        "",
        "| Condition | F1 | Precision | Recall | Accuracy | ROC-AUC | PR-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = [
        ("RGB-trained / RGB-tested", m_original),
        ("RGB-trained / grayscale-tested", m_testtime),
        ("Grayscale-trained / grayscale-tested", m_graydomain),
    ]
    for label, m in labels:
        lines.append(
            f"| {label} | {m['f1']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['accuracy']:.4f} | "
            f"{m['roc_auc']:.4f} | {m['pr_auc']:.4f} |"
        )
    lines += ["", "## Paired video/group bootstrap delta-F1", ""]
    for name, value in boot.items():
        g = value["video_group_delta_f1_ci95"]
        lines.append(
            f"- {name}: {value['observed_delta_f1']:+.4f} "
            f"(95% CI [{g[0]:+.4f}, {g[1]:+.4f}])"
        )
    (args.output_dir / f"{regime}_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    # Remove large caches from the result artifact; hashes remain in metadata.
    for path in (train_data, train_index, test_data, test_index):
        path.unlink(missing_ok=True)
    try:
        work.rmdir()
    except OSError:
        pass

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

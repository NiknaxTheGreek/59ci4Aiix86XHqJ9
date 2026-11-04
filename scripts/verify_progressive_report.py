"""Guard accepted MonReader progressive results against accidental report drift.

Runs offline with the Python standard library and NEVER recomputes model scores.
The source of accepted completed values is the 2026-11-03 final-results note.
"""
from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "results/progressive_temporal_completed_8_runs.csv"
EVIDENCE = ROOT / "results/progressive_study_provenance.md"
README = ROOT / "README.md"
NB_NAMES = [
    "updated_02_cnn_selection.ipynb",
    "updated_03_model_comparison.ipynb",
    "updated_04_generalization_and_stress_tests.ipynb",
    "updated_05_final_analysis.ipynb",
]
NUMERIC_COLUMNS = [
    "run_number", "coverage_percent", "n_train", "n_validation",
    "n_future", "n_future_sources", "future_macro_f1",
    "future_flip_recall", "false_negatives", "false_positives",
    "matched_macro_f1", "matched_flip_recall",
]


def verify():
    rows = list(csv.DictReader(TABLE.open(encoding="utf-8", newline="")))
    if len(rows) != 8:
        raise AssertionError(f"Expected 8 completed runs, got {len(rows)}")
    evidence = EVIDENCE.read_text(encoding="utf-8")
    evidence_rows = [
        line.split("|")[1:-1]
        for line in evidence.splitlines()
        if line.strip().startswith("| ") and line.split("|")[1].strip() in
        {str(i) for i in range(1, 9)}
    ]
    if len(evidence_rows) != 8:
        raise AssertionError("Evidence note no longer contains exactly 8 measured rows")

    evidence_cols = [
        "run_number", "regime", "coverage_percent", "n_train", "n_validation",
        "n_future", "n_future_sources", "future_macro_f1", "future_flip_recall",
        "false_negatives", "false_positives", "matched_macro_f1",
        "matched_flip_recall",
    ]
    for idx, (record, measured) in enumerate(zip(rows, evidence_rows), start=1):
        if len(measured) != len(evidence_cols):
            raise AssertionError(f"Run {idx}: unexpected evidence table width")
        values = dict(zip(evidence_cols, [x.strip().replace(",", "").replace("%", "") for x in measured]))
        if int(record["run_number"]) != idx or record["regime"] != ("T" if idx <= 4 else "ST"):
            raise AssertionError(f"Run {idx}: order/regime mismatch")
        if int(record["coverage_percent"]) != (25, 50, 75, 90)[(idx-1) % 4]:
            raise AssertionError(f"Run {idx}: coverage mismatch")
        for key in NUMERIC_COLUMNS:
            if Decimal(record[key]) != Decimal(values[key]):
                raise AssertionError(f"Run {idx}: {key} differs from measured source note")
        if int(record["n_future_sources"]) != (117 if idx <= 4 else 24):
            raise AssertionError(f"Run {idx}: wrong T/ST evaluation source population")
        if idx <= 4 and sum(int(record[k]) for k in ("n_train", "n_validation", "n_future")) != 2989:
            raise AssertionError(f"Run {idx}: incorrect T data accounting")
        for key in ("future_macro_f1", "future_flip_recall", "matched_macro_f1", "matched_flip_recall"):
            if not Decimal(0) <= Decimal(record[key]) <= Decimal(1):
                raise AssertionError(f"Run {idx}: invalid metric {key}")
        url = record["wandb_run_url"]
        if not url.startswith("https://forge.coreweave.com/wandb/") or url not in evidence:
            raise AssertionError(f"Run {idx}: W&B evidence URL missing or changed")

    readme = README.read_text(encoding="utf-8")
    for headline in ("## Does more labelled history improve page-flip detection", "## Business decision and next steps"):
        if headline not in readme:
            raise AssertionError(f"Missing README section: {headline}")
    for path in [
        ROOT / "figures/labelled_history_late_frame_macro_f1.svg",
        ROOT / "figures/evaluation_regimes.svg",
    ]:
        if not path.exists():
            raise AssertionError(f"Missing measured progressive figure {path}")
    for filename in NB_NAMES:
        notebook = json.loads((ROOT / filename).read_text(encoding="utf-8"))
        sources = [c.get("source", []) for c in notebook["cells"] if c["cell_type"] == "code"]
        code = "\n".join("".join(s) for s in sources)
        if "Setting RERUN_HISTORICAL_MODELS = False" not in code:
            raise AssertionError(f"{filename}: missing documented historical rerun mode")
        if "RERUN_HISTORICAL_MODELS = False" not in code:
            raise AssertionError(f"{filename}: historical report mode must default off")
        if filename.startswith("updated_04") and "RUN_NEW_PROGRESSIVE_TEMPORAL = False" not in code:
            raise AssertionError(f"{filename}: do not launch a new eight-run study by default")
    return True


if __name__ == "__main__":
    verify()
    print("PASS: completed eight-run figures/data/provenance and four report-first notebook switches")

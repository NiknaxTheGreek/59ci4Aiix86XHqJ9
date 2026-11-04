"""Static project checks: no dataset, credentials, or W&B account required."""
from __future__ import annotations
import ast
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "updated_01_data_and_splits.ipynb",
    "updated_02_cnn_selection.ipynb",
    "updated_03_model_comparison.ipynb",
    "updated_04_generalization_and_stress_tests.ipynb",
    "updated_05_final_analysis.ipynb",
]
MODULES = [
    "src/__init__.py", "src/common.py", "src/data.py", "src/splits.py",
    "src/models.py", "src/tuning.py", "src/evaluation.py",
    "src/visualization.py", "src/temporal.py",
]
FIGURES = [
    "figures/evaluation_regimes.svg",
    "figures/historical_performance_uncertainty.svg",
    "figures/rgb_vs_grayscale_examples.png",
    "figures/failure_analysis.svg",
    "figures/labelled_history_late_frame_macro_f1.svg",
    "figures/resnet18_gradcam_4x4.jpg",
]

def main():
    failures = []
    for rel in MODULES:
        path = ROOT / rel
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            failures.append(f"{rel}: {exc}")

    for rel in NOTEBOOKS:
        path = ROOT / rel
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            assert notebook["nbformat"] == 4
            assert isinstance(notebook["cells"], list) and notebook["cells"]
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    source = "".join(cell["source"])
                    if "%%writefile" in source or "MONREADER_SMOKE_TEST" in source:
                        failures.append(f"{rel} cell {index}: generated/smoke workflow not permitted")
                    compile(source, f"{rel}#cell{index}", "exec")
        except (OSError, ValueError, SyntaxError, AssertionError) as exc:
            failures.append(f"{rel}: {exc}")

    for rel in NOTEBOOKS:
        nb=json.loads((ROOT/rel).read_text(encoding="utf-8"))
        text="\n".join("".join(cell["source"]) for cell in nb["cells"])
        if "submission/results" in text or "submission'/'results" in text:
            failures.append(f"Old result directory referenced: {rel}")
    for rel in FIGURES:
        if not (ROOT / rel).is_file():
            failures.append(f"Missing published figure: {rel}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for rel in NOTEBOOKS:
        if rel not in readme:
            failures.append(f"README missing notebook reference: {rel}")
    if failures:
        print("\n".join(f"FAIL: {f}" for f in failures))
        return 1
    print("PASS: Python modules, five notebook code cells, README links, published figure paths")
    return 0

if __name__ == "__main__":
    sys.exit(main())

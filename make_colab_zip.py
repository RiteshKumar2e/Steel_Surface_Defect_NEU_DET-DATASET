#!/usr/bin/env python
"""Package the code, the frozen split and the already-measured baseline results
into one archive to upload to Colab. See COLAB.md for how to use it there.

The dataset is NOT included -- it ships separately as neu_data.zip, which
already exists in the project root.
"""
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "neu_colab.zip"

CODE = [
    "new_model_code.ipynb",             # the pipeline every script imports from
    "run_backbone_comparison.py",
    "run_multiseed.py",
    "make_corrected_tables.py",
    "make_module_figure.py",
    "COLAB.md",
    "paper_results/splits/split_v1.csv",
    "paper_results/splits/split_v1.sha256",
    "paper_results/tables/table_ablation.csv",
]


def collect():
    """(source path, archive name) pairs, archive names always POSIX."""
    items = []
    for rel in CODE:
        items.append((ROOT / rel, rel))
    # The baseline numbers that are already measured, so make_corrected_tables.py
    # runs immediately on Colab without retraining anything.
    base = ROOT / "paper_results" / "colab_results" / "results"
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.suffix in (".csv", ".json"):
            items.append((p, p.relative_to(ROOT).as_posix()))
    return items


def main():
    items = collect()
    missing = [a for s, a in items if not s.exists()]
    if missing:
        raise SystemExit("missing files:\n  " + "\n  ".join(missing))

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for src, arc in items:
            z.write(src, arc)

    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"{OUT.name}  --  {len(items)} files, {OUT.stat().st_size / 1024:.0f} KiB")
    print(f"sha256 {digest}\n")
    for arc in sorted(a for _, a in items):
        print("  ", arc)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Turn a ``train.py`` result pickle made with ``--save_weights`` into model files.

One file per fold, ``<out>/<name>_fold<k>.pt`` (``<out>/<name>.pt`` for a
single-fold run such as ``experiments/full_data.py``), each holding the
weights, the configuration and the normalisation constants
(``src/weights.py``), which ``scripts/simulate.py`` reads.

    python scripts/export_weights.py results/full_data/00_full_data.pkl --out results/weights --name full_data_seed00
    python scripts/export_weights.py results/settings/pub/00_pub.pkl --out results/weights/pub_seed00 --name pub
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from weights import export_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pickle")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--name", default=None, help="file name stem (default: the pickle's)")
    ap.add_argument("--variant", default="swa")
    a = ap.parse_args()
    for p in export_run(a.pickle, a.out, name=a.name, variant=a.variant):
        print(f"wrote {p} ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dimo.monitoring import read_committed, plot_curves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("e3", "dimo"), required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    rows = read_committed(a.log, a.stage)
    if not rows:
        raise RuntimeError("NO_COMMITTED_SCALARS")
    step = rows[-1]["global_step"]
    for kind in ("loss", "gradient"):
        plot_curves(rows, Path(a.output_dir) / f"{kind}_step_{step:06d}.png", stage=a.stage, step=step, kind=kind)


if __name__ == "__main__":
    main()

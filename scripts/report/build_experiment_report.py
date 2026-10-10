#!/usr/bin/env python3
"""Local lightweight evidence archive; never upload it or include model state."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dimo.monitoring import read_committed, plot_curves


def build_report(roots, output, *, stage="all", max_archive_bytes=64*1024**2):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dense-report-") as temporary:
        working = Path(temporary)
        summary = ["# Dense experiment report", "", "Unavailable measurements are NOT_AVAILABLE.", ""]
        index = {}
        for name, root in roots.items():
            if stage != "all" and name != stage:
                continue
            root = Path(root)
            folder = working / name; folder.mkdir()
            stage_root = root / "train" if name == "e3" and (root / "train").is_dir() else root
            log = stage_root / ("train_metrics.jsonl" if name == "e3" else "metrics.jsonl")
            rows = read_committed(log, name)
            summary.append(f"{name}: committed step {rows[-1]['global_step'] if rows else 'NOT_AVAILABLE'}")
            files = [log]
            for pattern in ("experiment_manifest.json", "run_provenance.json", "resolved_config.json", "parameter_report.json",
                            "disk_budget.json", "memory_rank*.jsonl", "failure_rank*.json", "runtime_acceptance_rank*.json"):
                files.extend(stage_root.glob(pattern))
            for directory in ("evaluation", "dev_evaluation"):
                evaluation = root / directory
                if evaluation.is_dir():
                    files.extend(evaluation.rglob("metrics.json"))
                    files.extend(evaluation.rglob("checkpoint_comparison.json"))
                    files.extend(evaluation.rglob("SELECTED_DENSE_CHECKPOINT.json"))
                    files.extend(evaluation.rglob("evaluation_identity.json"))
                    files.extend(evaluation.rglob("selection_preregistration.json"))
                    files.extend(evaluation.rglob("paired_to_teacher.json"))
                    files.extend(evaluation.rglob("*montage*.png"))
                    files.extend(evaluation.rglob("*fixed_cases.jpg"))
            for source in sorted(set(files)):
                if not source.is_file() or source.is_symlink():
                    continue
                relative = source.relative_to(root)
                target = folder / relative; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            if rows:
                for kind in ("loss", "gradient"):
                    plot_curves(rows, folder / f"{kind}.png", stage=name, step=rows[-1]["global_step"], kind=kind)
        (working / "summary.md").write_text("\n".join(summary)+"\n")
        for path in working.rglob("*"):
            if path.is_file():
                index[str(path.relative_to(working))] = hashlib.sha256(path.read_bytes()).hexdigest()
        (working / "archive_manifest.json").write_text(json.dumps(index, indent=2)+"\n")
        partial = output.with_suffix(output.suffix + ".partial")
        with tarfile.open(partial, "w:gz") as archive:
            for path in sorted(working.rglob("*")):
                if path.is_file():
                    archive.add(path, arcname=str(path.relative_to(working)))
        if partial.stat().st_size > max_archive_bytes:
            raise RuntimeError("REPORT_SIZE_LIMIT_EXCEEDED_PARTIAL_ARCHIVE_RETAINED")
        partial.rename(output)
    digest = hashlib.sha256()
    with output.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    output.with_suffix(output.suffix+".sha256").write_text(f"{digest.hexdigest()}  {output.name}\n")
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("e3", "dimo", "all"), required=True)
    p.add_argument("--e3-root")
    p.add_argument("--dimo-root")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    roots = {k: v for k, v in (("e3", a.e3_root), ("dimo", a.dimo_root)) if v}
    if not roots or (a.stage != "all" and a.stage not in roots):
        p.error("provide the requested stage root")
    print(build_report(roots, a.output, stage=a.stage))


if __name__ == "__main__":
    main()

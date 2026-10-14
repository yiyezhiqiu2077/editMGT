#!/usr/bin/env python3
import json
import os
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.train.verify_e3_dense_config import gpu_preflight
from src.explicit_region.config import load_config
from src.explicit_region.dense_runtime import validate_dense_config
from src.explicit_region.fixed_corpus import verify_mini_corpus_ready


def main():
    config = load_config("configs/train/dense_mini512_smoke.yaml")
    validate_dense_config(config)
    ready = verify_mini_corpus_ready(config["data"]["corpus_ready"], expected_rows=512)
    from collections import Counter
    counts = Counter(json.loads(line)["dataset_name"] for line in Path(config["data"]["manifest"]).read_text().splitlines() if line.strip())
    if counts != {"magicbrush": 256, "crispedit": 64, "scaleedit": 64, "interedit": 128}:
        raise RuntimeError("DENSE_MINI512_IDENTITY_COUNTS_MISMATCH")
    output = Path(os.environ["DENSE_OUTPUT_ROOT"])
    output.mkdir(parents=True, exist_ok=True)
    if any(p.name not in ("logs",) for p in output.iterdir()):
        raise RuntimeError("DENSE_LOCAL_OUTPUT_NOT_EMPTY")
    if shutil.disk_usage(output).free < 55_000_000_000:
        raise RuntimeError("DENSE_SMOKE_REQUIRES_AT_LEAST_55GB_FREE_DISK")
    gpu_preflight(config["dense"]["minimum_free_vram_gib"])
    print(json.dumps({"status": "READY_TO_RUN", "rows": 512, "dataset_counts": counts, "GPU_SMOKE": "NOT_RUN"}))


if __name__ == "__main__":
    main()

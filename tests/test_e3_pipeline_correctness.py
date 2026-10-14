import json
import random
import subprocess
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.data.prepare_magicbrush_assets import edit_mask_from_raw
from scripts.dev import download_mini_plan
from scripts.train.train_explicit_region import step_update_scheduler
from src.explicit_region.checkpoint import (
    capture_rank_rng_state, restore_rank_rng_state, validate_rank_rng_state,
)
from src.explicit_region.contracts import (
    audit_trainable_parameters, configure_region_trainability,
)
from src.explicit_region.fixed_corpus import verify_mini_corpus_ready
from src.explicit_region.fixed_dataset import RepeatedFixedCorpusDataset


ROOT = Path(__file__).resolve().parents[1]


def test_magicbrush_alpha_mask_polarity():
    rgba = Image.new("RGBA", (3, 1))
    rgba.putdata([(10, 20, 30, 0), (10, 20, 30, 255), (10, 20, 30, 128)])
    assert list(edit_mask_from_raw(rgba).getdata()) == [255, 0, 127]


class TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.ones(1))
        self.edit_region_embedding = torch.nn.Parameter(torch.ones(1))


def test_e3_region_embedding_is_trainable_and_audited():
    model = TinyTransformer()
    active = configure_region_trainability(
        model, {"mode": "roi_hardlock", "persistent_conditioning": True}
    )
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    report = audit_trainable_parameters(
        {"transformer": model}, optimizer, region_conditioning_active=active
    )
    assert active is True
    assert report["trainable_edit_region_embedding_params"] == 1


def test_inactive_region_embedding_is_excluded_from_optimizer():
    model = TinyTransformer()
    active = configure_region_trainability(
        model, {"mode": "full_target", "persistent_conditioning": False}
    )
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    report = audit_trainable_parameters(
        {"transformer": model}, optimizer, region_conditioning_active=active
    )
    assert active is False
    assert report["trainable_edit_region_embedding_params"] == 0


@pytest.mark.parametrize("sync,skipped,expected", [(False, False, 0), (True, True, 0), (True, False, 1)])
def test_scheduler_only_steps_after_successful_optimizer_boundary(sync, skipped, expected):
    class Scheduler:
        calls = 0
        def step(self):
            self.calls += 1
    scheduler = Scheduler()
    accelerator = SimpleNamespace(sync_gradients=sync, optimizer_step_was_skipped=skipped)
    assert step_update_scheduler(accelerator, scheduler) is bool(expected)
    assert scheduler.calls == expected


def test_single_rank_rng_bundle_roundtrip():
    random.seed(11); np.random.seed(12); torch.manual_seed(13)
    bundle = capture_rank_rng_state()
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    restore_rank_rng_state(bundle)
    actual = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert actual == expected
    validate_rank_rng_state(bundle, world_size=1)


def test_mini_corpus_marker_cannot_impersonate_formal(tmp_path):
    manifest = tmp_path / "train_512.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    from src.explicit_region.contracts import sha256_file
    marker = tmp_path / "MINI_CORPUS_READY.json"
    marker.write_text(json.dumps({
        "status": "READY", "schema_version": "mini-corpus-ready-v1",
        "dev_only": True, "total_rows": 512,
        "files": {"train": {"path": str(manifest), "sha256": sha256_file(manifest)}},
    }))
    assert verify_mini_corpus_ready(marker, expected_rows=512)["dev_only"] is True
    payload = json.loads(marker.read_text())
    payload["dev_only"] = False
    marker.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="MINI_CORPUS_NOT_READY"):
        verify_mini_corpus_ready(marker, expected_rows=512)


def test_repeated_fixed_corpus_preserves_base_identity_and_global_index():
    class Base:
        def __len__(self):
            return 3
        def set_epoch(self, epoch):
            self.epoch = epoch
        def get_with_global_index(self, index, global_index):
            return {"base_index": index, "global_sample_index": global_index}
    base = Base()
    repeated = RepeatedFixedCorpusDataset(base, 8)
    assert [repeated[i] for i in range(8)] == [
        {"base_index": i % 3, "global_sample_index": i} for i in range(8)
    ]
    repeated.set_epoch(4)
    assert base.epoch == 4


def test_partial_download_retries_and_resumes(monkeypatch):
    calls = []
    def flaky(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise ConnectionError("transient disconnect")
        return "/tmp/complete"
    monkeypatch.setattr(download_mini_plan, "hf_hub_download", flaky)
    monkeypatch.setattr(download_mini_plan.time, "sleep", lambda _seconds: None)
    assert download_mini_plan.download_with_retry(repo_id="owner/data", filename="large.tar") == "/tmp/complete"
    assert len(calls) == 3


def test_common_shell_short_tmpdir_and_parent_port_override():
    script = r'''
export EDITMGT_WORKTREE="$1"
export TMPDIR="/tmp/$(printf 'x%.0s' {1..100})"
export EDITMGT_MAIN_PROCESS_PORT=23456
source scripts/setup/common.sh
printf '%s\n%s\n%s\n' "$TMPDIR" "$TMP" "$(pick_main_process_port)"
'''
    result = subprocess.run(
        ["bash", "-c", script, "test", str(ROOT)], cwd=ROOT,
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    assert result[0].startswith("/tmp/editmgt-") and len(result[0]) < 80
    assert result[1] == result[0]
    assert result[2] == "23456"

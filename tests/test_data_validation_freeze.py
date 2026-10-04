import json
from pathlib import Path
import subprocess
import sys

import yaml

from scripts.data.freeze_fixed200k_validation import select_translated_aux
from src.explicit_region.canonical import sample_uid_for
from src.explicit_region.fixed_corpus import (
    MissingTranslation, freeze_aux_groups, validate_frozen_language,
)
from tests.fixed200k_helpers import synthetic_record


def han(row):
    row = dict(row, instruction_original="把物体改成红色", instruction_en="",
               translation_status="pending", language_original="zho_Hans")
    row["sample_uid"] = sample_uid_for(row)
    return row


def test_aux_pending_holds_candidate_then_qa_deterministically_backfills():
    rows = [han(synthetic_record("crispedit", i)) for i in range(4)]
    provisional = freeze_aux_groups(rows, "crispedit", 2)
    seen = []

    def missing(text, row):
        seen.append(row["sample_uid"])
        raise MissingTranslation(text)

    accepted, pending, rejected = select_translated_aux(rows, "crispedit", 2, 42, missing)
    assert accepted == rejected == []
    assert seen == [row["sample_uid"] for row in provisional]
    assert len(pending) == 2
    bad = provisional[0]["sample_uid"]

    def translate(text, row):
        return (text if row["sample_uid"] == bad else "Make it red"), "cache-key"

    selected, pending, rejected = select_translated_aux(rows, "crispedit", 2, 42, translate)
    repeated = select_translated_aux(list(reversed(rows)), "crispedit", 2, 42, translate)
    assert selected == repeated[0]
    assert pending == [] and len(rejected) == 1
    assert bad not in {row["sample_uid"] for row in selected}
    assert len({row["group_id"] for row in selected}) == 2
    for row in selected:
        validate_frozen_language(row)


def test_aux_requests_before_exhaustion_even_with_known_rejection():
    rows = [han(synthetic_record("crispedit", i)) for i in range(2)]
    first = freeze_aux_groups(rows, "crispedit", 2)[0]["sample_uid"]

    def mixed(text, row):
        if row["sample_uid"] == first:
            return text, "bad"
        raise MissingTranslation(text)

    _, pending, rejections = select_translated_aux(rows, "crispedit", 2, 42, mixed)
    assert len(pending) == len(rejections) == 1


def test_interedit_aux_qa_backfill_preserves_type_quota():
    rows = []
    for index, kind in enumerate(("Add", "Remove", "Local", "Texture")):
        canonical = "local_attribute" if kind == "Local" else kind.lower()
        rows.extend(han(synthetic_record("interedit", index * 10 + offset,
                                        original_type=kind, canonical_type=canonical)) for offset in range(3))
    from scripts.data.freeze_fixed200k_validation import freeze_interedit
    rejected_uid = freeze_interedit(rows, 4)[0]["sample_uid"]

    def translate(text, row):
        return (text if row["sample_uid"] == rejected_uid else "Make it red"), "key"

    selected, pending, rejections = select_translated_aux(rows, "interedit", 4, 42, translate)
    assert len(selected) == 4 and not pending and len(rejections) == 1
    assert {row["edit_type_original"] for row in selected} == {"Add", "Remove", "Local", "Texture"}


def test_freezer_cold_cache_does_not_publish_validation(tmp_path):
    config = yaml.safe_load(Path("configs/data/fixed_200k.yaml").read_text())
    config["translation"]["immutable_revision"] = "fixture-translation"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    args = [sys.executable, "scripts/data/freeze_fixed200k_validation.py", "--count", "1",
            "--config", str(config_path), "--output-dir", str(tmp_path / "validation"),
            "--translation-cache", str(tmp_path / "cache.jsonl")]
    for name in ("magicbrush", "crispedit", "scaleedit", "interedit"):
        row = synthetic_record(name, 0, original_type="Add", canonical_type="add")
        if name != "magicbrush":
            row = han(row)
        path = tmp_path / f"{name}.jsonl"; path.write_text(json.dumps(row) + "\n")
        args.extend(["--magicbrush-dev" if name == "magicbrush" else f"--{name}-pool", str(path)])
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 42, result.stderr
    assert not (tmp_path / "validation/validation.meta.json").exists()
    assert not (tmp_path / "validation/crispedit_aux128.jsonl").exists()
    pending = json.loads((tmp_path / "validation/translation_required.jsonl").read_text())
    assert pending["dataset"] == "crispedit"

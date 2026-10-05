import json
from pathlib import Path
import pytest
from PIL import Image
from torch.utils.data import DataLoader

from src.explicit_region.canonical import FrozenCorpusIntegrityError, sample_uid_for, validate_locator, validate_record, sha256_bytes
from src.explicit_region.fixed_dataset import FixedCorpusDataset
from tests.fixed200k_helpers import fixture_record


def test_four_dataset_contract_and_fixed_dispatch(tmp_path):
    rows=[];roots={}
    for i,name in enumerate(("magicbrush","crispedit","scaleedit","interedit")):
        root=tmp_path/name;roots[name]=root
        row=fixture_record(root,name,i,edit_type="add" if name=="interedit" else "other",original_type="Add" if name=="interedit" else "unknown")
        row["manifest_index"]=i;validate_record(row,require_frozen_index=True);rows.append(row)
    manifest=tmp_path/"train.jsonl";manifest.write_text("".join(json.dumps(row)+"\n" for row in rows))
    dataset=FixedCorpusDataset(manifest,roots,resolution=32)
    assert [dataset[i]["dataset_name"] for i in range(4)]==[row["dataset_name"] for row in rows]
    assert [dataset[i]["sample_uid"] for i in range(4)]==[row["sample_uid"] for row in rows]
    batch=next(iter(DataLoader(dataset,batch_size=4,shuffle=False)))
    assert batch["source_image"].shape==(4,3,32,32)
    assert list(batch["dataset_name"])==[row["dataset_name"] for row in rows]


def test_locator_rejects_absolute_and_parent_paths():
    with pytest.raises(ValueError): validate_locator({"backend":"file","relative_path":"/tmp/a"})
    with pytest.raises(ValueError): validate_locator({"backend":"tar","archive":"../a.tar","member":"x"})


def test_frozen_hash_mismatch_fails_without_replacement(tmp_path):
    root=tmp_path/"magic";row=fixture_record(root,"magicbrush",0);row["manifest_index"]=0
    manifest=tmp_path/"train.jsonl";manifest.write_text(json.dumps(row)+"\n")
    (root/row["source_locator"]["relative_path"]).write_bytes(b"corrupted")
    dataset=FixedCorpusDataset(manifest,{"magicbrush":root},resolution=32)
    with pytest.raises(FrozenCorpusIntegrityError,match="hash mismatch"): dataset[0]


def test_frozen_geometry_near_mismatch_still_fails_strict_runtime(tmp_path):
    root = tmp_path / "magicbrush"
    row = fixture_record(root, "magicbrush", 0)
    row["manifest_index"] = 0
    source_path = root / row["source_locator"]["relative_path"]
    target_path = root / row["target_locator"]["relative_path"]
    region_path = root / row["region_locator"]["relative_path"]
    Image.new("RGB", (1024, 1023), (10, 20, 30)).save(source_path)
    Image.new("RGB", (1024, 1024), (40, 50, 60)).save(target_path)
    Image.new("L", (1024, 1024), 255).save(region_path)
    row["source_sha256"] = sha256_bytes(source_path.read_bytes())
    row["target_sha256"] = sha256_bytes(target_path.read_bytes())
    row["region_sha256"] = sha256_bytes(region_path.read_bytes())
    row["sample_uid"] = sample_uid_for(row)
    validate_record(row, require_frozen_index=True)
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    dataset = FixedCorpusDataset(manifest, {"magicbrush": root}, resolution=32)
    with pytest.raises(FrozenCorpusIntegrityError, match="unaligned aspect ratio"):
        dataset[0]

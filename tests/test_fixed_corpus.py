import json
from pathlib import Path
import pytest
from torch.utils.data import DataLoader

from src.explicit_region.canonical import FrozenCorpusIntegrityError, validate_locator, validate_record
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

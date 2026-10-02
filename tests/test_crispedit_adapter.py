from src.explicit_region.crispedit import CrispEditAlignedDataset
from src.explicit_region.canonical import validate_record
from tests.fixed200k_helpers import fixture_record


def test_crispedit_canonical_shared_geometry(tmp_path):
    row=fixture_record(tmp_path,"crispedit",1,edit_type="replace",original_type="replacement")
    validate_record(row);dataset=CrispEditAlignedDataset([row],tmp_path,resolution=32)
    item=dataset[0]
    assert item["dataset_name"]=="crispedit"
    assert item["source_image"].shape==item["target_image"].shape==(3,32,32)
    assert item["edit_region_mask"].shape==(32,32) and item["edit_region_mask"].any()
    assert item["sample_uid"]==row["sample_uid"]

from src.explicit_region.scaleedit import ScaleEditAlignedDataset
from tests.fixed200k_helpers import fixture_record


def test_scaleedit_canonical_shared_geometry(tmp_path):
    row=fixture_record(tmp_path,"scaleedit",2,edit_type="local_attribute",original_type="color")
    item=ScaleEditAlignedDataset([row],tmp_path,resolution=32)[0]
    assert item["dataset_name"]=="scaleedit"
    assert item["source_image"].shape==item["target_image"].shape==(3,32,32)
    assert item["edit_region_mask"].any()

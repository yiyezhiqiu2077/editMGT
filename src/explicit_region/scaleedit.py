"""ScaleEdit canonical adapter; raw schema mapping is audit-driven."""

from .fixed_dataset import CanonicalAlignedDataset


class ScaleEditAlignedDataset(CanonicalAlignedDataset):
    expected_dataset_name = "scaleedit"

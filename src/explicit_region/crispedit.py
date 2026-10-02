"""CrispEdit canonical adapter; raw schema mapping is audit-driven."""

from .fixed_dataset import CanonicalAlignedDataset


class CrispEditAlignedDataset(CanonicalAlignedDataset):
    expected_dataset_name = "crispedit"

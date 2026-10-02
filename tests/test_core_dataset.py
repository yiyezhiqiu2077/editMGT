from src.explicit_region.dataset import DeterministicCoreDataset


class FakeDataset:
    def __init__(self, name, length=11):
        self.name, self.length = name, length

    def __len__(self):
        return self.length

    def get_with_global_index(self, index, global_index):
        return {"dataset_name": self.name, "index": index, "global_sample_index": global_index}


class PartlyBrokenDataset(FakeDataset):
    def get_with_global_index(self, index, global_index):
        if index % 3:
            raise FileNotFoundError("fixture missing")
        return super().get_with_global_index(index, global_index)


def test_core_sampler_is_stateless_and_near_preregistered_weight():
    dataset = DeterministicCoreDataset(
        FakeDataset("interedit"), FakeDataset("magicbrush"),
        interedit_weight=0.8, length=10_000, base_seed=42,
    )
    first = [dataset[i] for i in range(100)]
    assert first == [dataset[i] for i in range(100)]
    fraction = sum(dataset[i]["dataset_name"] == "interedit" for i in range(10_000)) / 10_000
    assert 0.78 < fraction < 0.82
    assert all(dataset[i]["global_sample_index"] == i for i in range(100))


def test_replacement_is_deterministic_for_fresh_and_resume():
    def build():
        return DeterministicCoreDataset(
            PartlyBrokenDataset("interedit", 13), PartlyBrokenDataset("magicbrush", 13),
            interedit_weight=0.8, length=100, base_seed=7, max_replacement_attempts=32,
        )
    fresh = [build()[g] for g in range(20)]
    resumed = [build()[g] for g in range(10, 20)]
    assert fresh[10:] == resumed
    assert any(item["replacement_attempt"] > 0 for item in fresh)

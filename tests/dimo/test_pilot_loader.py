import json

import pytest
import torch
from torch.utils.data import Dataset

from src.dimo.pilot import PilotEpochLoader
from src.dimo.pilot_contract import validate_output_prefix
from src.explicit_region.epoch_sampler import DeterministicEpochSampler


class FixtureDataset(Dataset):
    def __len__(self):
        return 64

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        return {'index': index, 'epoch': self.epoch}


def test_worker_rng_and_loader_suffix_do_not_change_process_rng():
    before = torch.get_rng_state().clone()
    for rank in range(8):
        def loader(cursor):
            sampler = DeterministicEpochSampler(64, base_seed=42, epoch=0,
                train_manifest_sha256='fixture', rank=rank, world_size=8,
                samples_consumed_in_epoch=cursor)
            return PilotEpochLoader(FixtureDataset(), sampler, {'epochs': 1, 'seed': 42})
        full = [int(row['index'][0]) for row in loader(0)]
        resumed = [int(row['index'][0]) for row in loader(16)]
        assert resumed == full[2:]
    assert torch.equal(before, torch.get_rng_state())


def test_resume_output_refuses_stale_checkpoint_or_unrelated_directory(tmp_path):
    validate_output_prefix(tmp_path, resume_state=None)
    (tmp_path / 'unrelated.txt').write_text('user data')
    with pytest.raises(RuntimeError, match='PREFIX_INCOMPLETE'):
        validate_output_prefix(tmp_path, resume_state={'global_dimo_step': 10}, config_hash='h')
    (tmp_path / 'run_manifest.json').write_text(json.dumps({'config_hash': 'h'}))
    (tmp_path / 'metrics.jsonl').write_text(''.join(json.dumps({'global_dimo_step': step}) + '\n'
                                               for step in range(1, 11)))
    validate_output_prefix(tmp_path, resume_state={'global_dimo_step': 10}, config_hash='h')
    with pytest.raises(RuntimeError, match='STEP_MISMATCH'):
        validate_output_prefix(tmp_path, resume_state={'global_dimo_step': 5}, config_hash='h')
    with pytest.raises(RuntimeError, match='CONFIG_IDENTITY'):
        validate_output_prefix(tmp_path, resume_state={'global_dimo_step': 10}, config_hash='different')

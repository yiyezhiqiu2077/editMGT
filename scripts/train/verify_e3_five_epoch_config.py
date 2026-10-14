"""Check the E3-only budget and inherited unchanged recipe before launch."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.explicit_region.config import _load_unexpanded


def verify(config):
    expected = {'experiment': 'E3-D200K-5Epoch', 'epochs': 5,
                'max_optimizer_steps': 31250, 'scheduler_horizon_steps': 31250,
                'warmup_steps': 200, 'resolution': 1024, 'batch_per_gpu': 1,
                'gradient_accumulation': 4, 'checkpoint_steps': list(range(3125, 31251, 3125))}
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f'E3_FIVE_EPOCH_CONTRACT_MISMATCH: {key}')
    if config['data']['name'] != 'fixed_200k' or config['data'].get('repeat_to_optimizer_steps', False):
        raise RuntimeError('E3 requires real no-replacement fixed-200k epochs')
    baseline = _load_unexpanded(ROOT / 'configs/train/_cluster_8g_base.yaml')
    for key in ['corruption', 'lora', 'condition_dropout', 'token_mask', 'geometry', 'optimizer']:
        if config[key] != baseline[key]:
            raise RuntimeError(f'E3_CORE_RECIPE_CHANGED: {key}')
    if (config['corruption']['mode'] != 'roi_hardlock'
            or config['corruption']['persistent_conditioning'] is not True
            or config['lora']['scope'] != 'both'
            or config['optimizer']['learning_rate'] != 3e-5):
        raise RuntimeError('E3_CORE_RECIPE_CHANGED')
    return {**expected, 'expected_dataset_rows': 200000, 'steps_per_epoch': 6250,
            'sample_exposures': 1000000, 'GPUs': 8, 'global_batch': 32,
            'learning_rate': 3e-5, 'corruption': config['corruption'], 'lora_scope': 'both',
            'diagnostic_only_CFG': 5}


if __name__ == '__main__':
    print(json.dumps(verify(_load_unexpanded(ROOT / 'configs/train/cluster_8g_e3.yaml')), indent=2))

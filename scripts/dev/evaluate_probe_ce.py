"""Measure checkpoint CE on identical frozen samples and corruption seeds."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig
from torch.utils.data import DataLoader

from scripts.train.train_explicit_region import prepare_batch
from src.explicit_region.checkpoint import load_trainable_state
from src.explicit_region.config import load_config
from src.explicit_region.fixed_dataset import FixedCorpusDataset
from src.explicit_region.modeling import load_released_components


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--full-roi', action='store_true',
                   help='Evaluation-only: mask the entire edit ROI, with no target context.')
    a = p.parse_args()
    c = load_config(a.config)
    if a.full_roi:
        if c['corruption']['mode'] != 'roi_hardlock':
            raise ValueError('Full-ROI probe requires the unchanged ROI-hardlock recipe')
        c['corruption'] = dict(c['corruption'], full_roi_mask_probability=1.0)
    set_seed(c['seed'])
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    accelerator = Accelerator(mixed_precision=c['precision'])
    components = load_released_components(c['model']['repo_or_root'], torch_dtype=torch.bfloat16, vq_dtype=torch.float32)
    components.transformer.add_adapter(LoraConfig(
        r=c['lora']['rank'], lora_alpha=c['lora']['alpha'],
        lora_dropout=c['lora']['dropout'], target_modules=c['lora']['target_modules'],
    ))
    load_trainable_state(components.transformer, a.checkpoint)
    for model in (components.transformer, components.text_encoder, components.llm_encoder, components.vqvae):
        model.to(accelerator.device).eval().requires_grad_(False)
    data = c['data']
    dataset = FixedCorpusDataset(data['manifest'], data['roots'],
        resolution=c['resolution'], base_seed=c['seed'], verify_hashes=True,
        max_random_attempts=c['geometry']['max_resample_attempts'],
        minimum_mask_retention=c['geometry']['minimum_mask_retention'])
    rows = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=1, num_workers=0):
            with accelerator.autocast():
                loss, _, corruption, _ = prepare_batch(batch, components, accelerator, c, scope=c['lora']['scope'])
            if a.full_roi:
                assert torch.equal(corruption.selected_mask, corruption.edit_region_mask.bool())
                assert torch.equal(corruption.labels.ne(-100), corruption.selected_mask)
                assert bool(corruption.full_roi_branch.all())
                assert bool(corruption.input_tokens[corruption.selected_mask].eq(
                    components.transformer.config.vocab_size - 1).all())
            value = float(loss)
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite probe CE')
            rows.append({'sample_uid': batch['sample_uid'][0], 'ce': value,
                         'valid_tokens': int(corruption.labels.ne(-100).sum())})
    result = {'checkpoint': str(Path(a.checkpoint).resolve()), 'samples': len(rows),
              'ce': sum(row['ce'] for row in rows) / len(rows),
              'protocol': 'fixed-manifest-index-corruption-eval-mode',
              'full_roi_probe': a.full_roi, 'per_sample': rows}
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'ce': result['ce'], 'samples': len(rows), 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()

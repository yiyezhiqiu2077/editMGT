"""Authorized budget extension with strict same-recipe full-state transfer."""
import copy
import json
from pathlib import Path
from src.explicit_region.checkpoint import recipe_fingerprint
from src.explicit_region.contracts import sha256_file
from .reward_fork import protect_outputs


def continuation_manifest(source, authorization):
    source, authorization = Path(source).resolve(), Path(authorization).resolve()
    marker = json.loads((source / 'COMPLETED.json').read_text())
    identity = json.loads((source / 'inference_identity.json').read_text())
    record = json.loads(authorization.read_text())
    config = source.parent.parent / 'resolved_config.json'
    if (marker.get('schema') != 'full-dense-dimo-ddp-v3'
            or not identity.get('reward_extension')
            or 'training_state.pt' not in marker['files']
            or record['selected_complete_resume_checkpoint'] != str(source)):
        raise RuntimeError('SOFT_REWARD_CONTINUATION_REQUIRES_SELECTED_FULL_REWARD_CHECKPOINT')
    return {'schema': 'soft-reward-authorized-long-v1', 'source': str(source),
            'source_step': marker['committed_step'],
            'completed_sha256': sha256_file(source / 'COMPLETED.json'),
            'identity_sha256': sha256_file(source / 'inference_identity.json'),
            'source_config_sha256': sha256_file(config),
            'authorization': str(authorization), 'authorization_sha256': sha256_file(authorization),
            'target_step': record['long_target_step'], 'checkpoint_interval': record['checkpoint_frequency'],
            'state_policy': 'preserve_roles_optimizers_schedulers_ema_rng_sampler_reward_calibration'}


def training_recipe(config):
    value = copy.deepcopy(config)
    # Only paths, diagnostics, and the explicit operational authorization may
    # change. Data budget, optimizer, RNG, and all reward settings remain exact.
    for name in ('output_dir', 'soft_reward_continuation', 'full_update_diagnostic_steps'):
        value.pop(name, None)
    value['soft_reward'].pop('inference_output_dir', None)
    return value


def validate_continuation(config, new_identity=None):
    manifest = config['soft_reward_continuation']
    if continuation_manifest(manifest['source'], manifest['authorization']) != manifest:
        raise RuntimeError('SOFT_REWARD_CONTINUATION_IDENTITY_OR_AUTHORIZATION_CHANGED')
    root = Path(manifest['source'])
    parent = json.loads((root / 'inference_identity.json').read_text())
    old_config = json.loads((root.parent.parent / 'resolved_config.json').read_text())
    if (recipe_fingerprint(old_config) != parent['config_sha256']
            or training_recipe(old_config) != training_recipe(config)
            or manifest['target_step'] != config['max_optimizer_steps']
            or manifest['checkpoint_interval'] != config['checkpoint_interval_steps']
            or manifest['checkpoint_interval'] != 2000):
        raise RuntimeError('SOFT_REWARD_CONTINUATION_TRAINING_RECIPE_CHANGED')
    protect_outputs(config['output_dir'], root)
    protect_outputs(config['soft_reward']['inference_output_dir'], root)
    if new_identity is not None:
        ignore = {'config_sha256', 'git_sha', 'reward_extension', 'reward_continuation'}
        if ({k:v for k,v in parent.items() if k not in ignore}
                != {k:v for k,v in new_identity.items() if k not in ignore}):
            raise RuntimeError('SOFT_REWARD_CONTINUATION_MODEL_OR_DATA_IDENTITY_CHANGED')
        old_reward = copy.deepcopy(parent['reward_extension'])
        new_reward = copy.deepcopy(new_identity['reward_extension'])
        old_reward.pop('inference_output_dir', None)
        new_reward.pop('inference_output_dir', None)
        if old_reward != new_reward:
            raise RuntimeError('SOFT_REWARD_CONTINUATION_REWARD_RECIPE_CHANGED')
    return parent


def load_continuation(source, *, config, new_identity, **state_objects):
    if Path(source).resolve() != Path(config['soft_reward_continuation']['source']).resolve():
        raise RuntimeError('SOFT_REWARD_CONTINUATION_SOURCE_MISMATCH')
    parent = validate_continuation(config, new_identity)
    # Verify and restore against the original identity. Future checkpoints
    # carry the new execution identity plus immutable full-state ancestry.
    from .dense_distributed_checkpoint import load_checkpoint
    return load_checkpoint(source, expected_identity=parent,
                           expected_config_hash=parent['config_sha256'], **state_objects)

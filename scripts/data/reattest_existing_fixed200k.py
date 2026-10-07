"""Re-attest existing frozen data only; never downloads or rebuilds a corpus."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.explicit_region.fixed_corpus import verify_corpus_ready
from src.explicit_region.formal_pipeline import (
    git_sha, identity_path, load_formal_assets, sha256_file, stable_json_hash,
    validate_resolved_revision, verify_formal_asset_corpus, write_json_atomic,
)
from src.explicit_region.reattestation import (
    DATASETS, alpha_probe, discover_unique, frozen_identity, recorded_environment,
    verify_current_loader, publish_ready,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-root', required=True)
    p.add_argument('--asset-root', required=True)
    p.add_argument('--model-root', required=True)
    a = p.parse_args()
    run, assets_root, model = [Path(v).resolve() for v in (a.run_root, a.asset_root, a.model_root)]
    if any(not path.is_dir() for path in [run, assets_root, model]):
        raise RuntimeError('REATTTEST_ROOT_MISSING')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT).strip():
        raise RuntimeError('CURRENT_CODE_NOT_CLEAN: checkout the exact pushed SHA first')
    marker = discover_unique(run, 'CORPUS_READY.json')
    ready = verify_corpus_ready(marker)
    if ready.get('schema_version') != 'fixed200k-ready-v3':
        raise RuntimeError('REATTTEST_UNSUPPORTED_OLD_READY: v3 hash-bound reports required')
    ledgers = sorted(set(run.rglob('formal_assets.json')) | set(assets_root.rglob('formal_assets.json')))
    matching = [path for path in ledgers if json.loads(path.read_text()).get('formal_identity_sha256') == ready['formal_assets_sha256']]
    if len(matching) != 1:
        raise RuntimeError('FROZEN_FORMAL_ASSET_IDENTITY_MISSING_OR_AMBIGUOUS')
    ledger_path = matching[0]
    old_assets = json.loads(ledger_path.read_text())
    if old_assets.get('corpus_ready', {}).get('sha256') != sha256_file(marker):
        raise RuntimeError('OLD_FORMAL_ASSETS_CORPUS_HASH_MISMATCH')
    env_paths = sorted(set(run.rglob('formal_env.sh')) | set(assets_root.rglob('formal_env.sh')))
    if len(env_paths) != 1:
        raise RuntimeError('CANONICAL_ROOTS_ENV_RECORD_MISSING_OR_AMBIGUOUS')
    env = recorded_environment(env_paths[0])
    roots = {name: Path(env[name.upper() + '_ROOT']) for name in DATASETS}
    pinned = load_formal_assets(ROOT / 'configs/formal_assets.yaml')
    for name, record in old_assets['assets'].items():
        spec = pinned['assets'][name]
        if spec['provider'] == 'huggingface':
            validate_resolved_revision(spec['revision'], record['resolved_revision'])
            identity = json.loads(identity_path(record['local_path']).read_text())
            if identity.get('resolved_revision') != record['resolved_revision'] or identity.get('repo_id') != spec['repo_id']:
                raise RuntimeError('FORMAL_ASSET_IDENTITY_MISMATCH')
    if Path(old_assets['assets']['editmgt']['local_path']).resolve() != model:
        raise RuntimeError('EDITMGT_MODEL_ROOT_IDENTITY_MISMATCH')
    before, rows, validation = frozen_identity(ready)
    if set(before['dataset_counts']) != set(DATASETS) or before['dataset_counts'] != ready['dataset_counts']:
        raise RuntimeError('FROZEN_DATASET_ALLOCATION_MISMATCH')
    for name in ['translation_report', 'machine_qa', 'schema_contract_report', 'duplicate_report', 'audit_report', 'translation_cache']:
        if name not in ready['files']:
            raise RuntimeError(f'REATTTEST_REQUIRED_REPORT_MISSING: {name}')
    for name in ['machine_qa', 'schema_contract_report']:
        if json.loads(Path(ready['files'][name]['path']).read_text()).get('status') != 'PASS':
            raise RuntimeError(f'REATTTEST_OLD_REPORT_NOT_PASS: {name}')
    schemas = {path.stem: sha256_file(path) for path in sorted((ROOT / 'configs/data/schema').glob('*.yaml'))}
    if schemas != old_assets.get('schema_contract_hashes'):
        raise RuntimeError('SCHEMA_CONTRACT_CHANGED: full schema revalidation required; no corpus rewrite allowed')
    for row in rows:
        if row['dataset_revision'] != old_assets['assets'][row['dataset_name']]['resolved_revision']:
            raise RuntimeError('FROZEN_DATASET_REVISION_MISMATCH')
    current = git_sha(ROOT)
    output = Path(tempfile.mkdtemp(prefix=f'reattest-{current[:12]}-', dir=marker.parent))
    shutil.copyfile(marker, output / 'CORPUS_READY.previous.json')
    shutil.copyfile(ledger_path, output / 'formal_assets.previous.json')
    write_json_atomic(output / 'identity.before.json', before)
    alpha = alpha_probe(rows, roots['magicbrush'], old_assets['assets']['magicbrush'], ready)
    write_json_atomic(output / 'MAGICBRUSH_ALPHA_PROBE.json', alpha)
    loader = verify_current_loader(ready['files']['train_200k']['path'], rows, validation,
                                   roots, Path(env['MAGICBRUSH_DEV_ROOT']))
    write_json_atomic(output / 'current_loader_compatibility.json', loader)
    after, _, _ = frozen_identity(ready)
    if before != after:
        raise RuntimeError('FROZEN_CONTENT_CHANGED_DURING_REATTESTATION')
    # Old pure-data results are reused only after all their READY-bound hashes
    # and every actual asset/geometry have been rechecked by production loaders.
    verify_corpus_ready(marker)
    write_json_atomic(output / 'identity.after.json', after)
    new_assets = dict(old_assets, git_sha=current)
    new_assets.pop('corpus_ready', None)
    new_assets.pop('formal_identity_sha256', None)
    new_assets['formal_identity_sha256'] = stable_json_hash(new_assets)
    new_ready = dict(ready, git_sha=current, formal_assets_sha256=new_assets['formal_identity_sha256'])
    new_ready['files'] = dict(ready['files'])
    for name, file in [('magicbrush_alpha_probe', output / 'MAGICBRUSH_ALPHA_PROBE.json'),
                       ('current_loader_compatibility', output / 'current_loader_compatibility.json'),
                       ('reattest_identity_before', output / 'identity.before.json'),
                       ('reattest_identity_after', output / 'identity.after.json')]:
        new_ready['files'][name] = {'path': str(file), 'sha256': sha256_file(file)}
    new_ready['reattestation'] = {'status': 'READY', 'git_sha': current,
                                 'manifest_unchanged': True, 'validation_unchanged': True,
                                 'alpha_mode': alpha['mode'], 'loader_rows': loader['rows']}
    candidate = output / 'CORPUS_READY.candidate.json'
    write_json_atomic(candidate, new_ready)
    verify_corpus_ready(candidate)
    new_assets['corpus_ready'] = {'path': str(marker), 'sha256': sha256_file(candidate)}
    # Never replace old READY before every check passes. Retain backups for
    # recovery; if the second atomic replacement fails, restore both originals.
    publish_ready(marker, ledger_path, new_ready, new_assets, current, output)
    print('\n'.join(['EXISTING_D200K_ROWS=200000', 'MANIFEST_UNCHANGED=YES',
                     'VALIDATION_UNCHANGED=YES', 'MAGICBRUSH_ALPHA_MASK=PASS',
                     f"MAGICBRUSH_ALPHA_PROBE_MODE={alpha['mode']}", 'CURRENT_CODE_COMPATIBLE=YES',
                     'CURRENT_GIT_SHA_BOUND=YES', 'CORPUS_REATTESTED=YES']), flush=True)


if __name__ == '__main__':
    main()

import torch

from src.dimo.checkpoint import load_dimo_checkpoint, save_dimo_checkpoint
from src.dimo.rng import RNG_NAMESPACES, stable_seed
from src.dimo.step import complete_dimo_step
from tests.dimo.toy import batch, make_bundle, step_config


def run_step(bundle, step, uid):
    roles, student_opt, aux_opt, student_sched, aux_sched, ema = bundle
    return complete_dimo_step(
        roles, **batch(uid), student_optimizer=student_opt, auxiliary_optimizer=aux_opt,
        student_scheduler=student_sched, auxiliary_scheduler=aux_sched, student_ema=ema,
        base_seed=77, epoch=0, committed_dimo_step=step, mask_token_id=5,
        codebook_size=5, config=step_config(),
    )


def test_rng_namespaces_are_stable_and_independent():
    seeds = [stable_seed(1, 2, "uid", 3, name) for name in RNG_NAMESPACES]
    assert seeds == [stable_seed(1, 2, "uid", 3, name) for name in RNG_NAMESPACES]
    assert len(set(seeds)) == len(seeds)


def test_two_steps_equal_one_plus_resume(tmp_path):
    uninterrupted = make_bundle()
    run_step(uninterrupted, 0, "sample-a")
    full_diag, full_artifacts = run_step(uninterrupted, 1, "sample-b")

    first_leg = make_bundle()
    run_step(first_leg, 0, "sample-a")
    roles, student_opt, aux_opt, student_sched, aux_sched, ema = first_leg
    save_dimo_checkpoint(
        tmp_path, roles=roles, student_optimizer=student_opt, auxiliary_optimizer=aux_opt,
        student_scheduler=student_sched, auxiliary_scheduler=aux_sched, student_ema=ema,
        committed_dimo_step=1, fixed_corpus_cursor=1, epoch=0, samples_consumed=1,
        config_sha256="config", teacher_checkpoint_hash="teacher", upstream_commit="upstream",
    )

    resumed = make_bundle()
    roles, student_opt, aux_opt, student_sched, aux_sched, ema = resumed
    state = load_dimo_checkpoint(
        tmp_path, roles=roles, student_optimizer=student_opt, auxiliary_optimizer=aux_opt,
        student_scheduler=student_sched, auxiliary_scheduler=aux_sched, student_ema=ema,
        expected_config_hash="config", expected_teacher_hash="teacher",
        expected_upstream_commit="upstream",
    )
    assert state["global_dimo_step"] == 1
    assert state["fixed_corpus_cursor"] == state["samples_consumed"] == 1
    resumed_diag, resumed_artifacts = run_step(resumed, 1, "sample-b")

    for key in (
        "initial_mask", "initial_tokens", "student_tokens",
        "teacher_pseudo_mask", "auxiliary_pseudo_mask", "embedding_noise",
    ):
        assert torch.equal(full_artifacts[key], resumed_artifacts[key])
    assert full_diag["loss_dimo"] == resumed_diag["loss_dimo"]
    assert full_diag["loss_aux"] == resumed_diag["loss_aux"]
    for role in ("student", "auxiliary"):
        full_state = uninterrupted[0].role_state_dict(role)
        resumed_state = resumed[0].role_state_dict(role)
        assert full_state.keys() == resumed_state.keys()
        for key in full_state:
            assert torch.equal(full_state[key], resumed_state[key])
    assert uninterrupted[-1].state_dict()["shadow"].keys() == resumed[-1].state_dict()["shadow"].keys()
    for key, value in uninterrupted[-1].state_dict()["shadow"].items():
        assert torch.equal(value, resumed[-1].state_dict()["shadow"][key])

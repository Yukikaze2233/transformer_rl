"""Fresh guarded CPU learning and explicit complete-state stage transitions."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import random

import numpy as np
import pytest
import torch

from transformer_rl.frame_checkpoint import capture_rng, load_frame_checkpoint, restore_rng
from transformer_rl.frame_config import digest
from transformer_rl.frame_continuation import FrameContinuation
from transformer_rl.frame_training import FrameActorCritic, FrameCollector
from transformer_rl.frame_workflow import _model_state_sha256, _seed
from transformer_rl.retention import save_anchors
from test_frame_continuation import assert_same, configuration, make_env, parent


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def initial_model(config, seed=71):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        with torch.device("cpu"):
            model = FrameActorCritic(config.model)
        rng = torch.get_rng_state().clone()
    return model, rng


def start(config, factory=make_env, **overrides):
    options = dict(rollout_steps=4, training_seed=71, retention_seed=901,
        evaluation_seeds=(801, 802),
        expected_initial_model_sha256=_model_state_sha256(initial_model(config)[0]))
    options.update(overrides)
    return FrameContinuation.start(config, factory, "packed_env:make_env", **options)


def resume(config, path, receipt, *, update, transitions, factory=make_env, **overrides):
    options = dict(checkpoint_sha256=receipt["sha256"], parent_update=update,
        cumulative_transitions=transitions, consumed_updates=update,
        rollout_steps=4, training_seed=71, retention_seed=901,
        evaluation_seeds=(801, 802), resume=True)
    options.update(overrides)
    return FrameContinuation.open(config, factory, "packed_env:make_env", path, **options)


def noisy_factory(events):
    def factory(**kwargs):
        torch.rand(19)
        random.random()
        np.random.rand(7)
        env = make_env(**kwargs)
        env.set_training_progress = lambda updates, transitions: events.append(("progress", updates, transitions))
        original = env.reset
        def reset(seed=None):
            events.append(("reset", seed, kwargs["environment_config"].get("stage", "fresh")))
            torch.rand(23)
            random.random()
            np.random.rand(11)
            return original(seed)
        env.reset = reset
        return env
    return factory


@pytest.mark.parametrize("architecture,readout", [
    ("mlp", "last"), ("history_mlp", "last"),
    ("transformer", "last"), ("transformer", "query"),
])
def test_start_binds_initial_weights_zero_clock_and_rng_after_noisy_reset(tmp_path, architecture, readout):
    config = configuration(architecture, readout)
    expected_model, expected_cpu_rng = initial_model(config)
    expected_hash = _model_state_sha256(expected_model)
    events = []
    with start(config, noisy_factory(events)) as session:
        assert events == [("progress", 0, 0), ("reset", 71, "fresh")]
        assert_same(session.model.state_dict(), expected_model.state_dict())
        assert session.update == session.consumed_updates == session.collected_transitions == 0
        assert session.trainer.optimizer.state_dict()["state"] == {}
        assert session.sampler.coefficient == 0 and session.sampler.identities == []
        assert session.metadata["initialization_guard"] == {
            "expected_sha256": expected_hash, "actual_sha256": expected_hash, "verified": True}
        observed = capture_rng()
        assert torch.equal(observed["torch"], expected_cpu_rng)
        assert_same(observed["python"], random.Random(71).getstate())
        name, values, *rest = np.random.RandomState(71).get_state()
        assert_same(observed["numpy"], [name, values.tolist(), *rest])
        initial_sampler = session.sampler.state_dict()
        assert session.sampler().item() == 0
        assert_same(session.sampler.state_dict(), initial_sampler)
        assert_same(capture_rng(), observed)
        path = tmp_path / "zero.pt"
        receipt = session.save(path)
    model, trainer, recovered, update, metadata, saved_rng = load_frame_checkpoint(path)
    assert recovered.to_dict() == config.to_dict() and update == 0
    assert metadata["collected_transitions"] == 0
    assert metadata["continuation"]["clock"] == {
        "consumed_updates": 0, "collected_transitions": 0, "rollout_steps": 4}
    assert metadata["continuation_parent"] is None
    assert metadata["source"] and metadata["environment_provenance"]["identity"]
    assert metadata["initial_rng"]["algorithm"] == "seeded_global_learning_after_environment_reset_v1"
    assert_same(saved_rng, observed)
    assert_same(model.state_dict(), expected_model.state_dict())
    assert trainer.optimizer.state_dict()["state"] == {}
    assert receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with resume(config, path, receipt, update=0, transitions=0) as recovered_session:
        assert_same(recovered_session.sampler.state_dict(), initial_sampler)
        assert_same(capture_rng(), observed)


@pytest.mark.parametrize("architecture,readout", [
    ("mlp", "last"), ("history_mlp", "last"),
    ("transformer", "last"), ("transformer", "query"),
])
def test_three_environment_segments_retain_actual_adam_global_private_rng_and_clocks(
        tmp_path, architecture, readout):
    base = configuration(architecture, readout)
    events = []
    previous_path = previous_receipt = None
    previous_model = previous_adam = previous_rng = previous_sampler = None
    for stage in range(3):
        config = replace(base, environment={**base.environment, "stage": stage})
        factory = noisy_factory(events)
        if stage == 0:
            session = start(config, factory)
        else:
            session = resume(config, previous_path, previous_receipt,
                update=stage * 2, transitions=stage * 24,
                factory=factory, environment_transition=True)
            assert_same(session.model.state_dict(), previous_model)
            assert_same(session.trainer.optimizer.state_dict(), previous_adam)
            assert_same(capture_rng(), previous_rng)
            assert_same(session.sampler.state_dict(), previous_sampler)
            assert set(session.metadata["continuation_parent"]) == {"path", "sha256", "update", "resume"}
            assert session.metadata["stage_transition"] == {
                "environment_transition": True,
                "parent_environment_sha256": digest({**base.environment, "stage": stage - 1}),
                "environment_sha256": digest(config.environment)}
        assert events[-2:] == [("progress", stage * 2, stage * 24), ("reset", 71, stage)]
        assert session.metadata["history_reset"] == "repeat_first"
        assert session.metadata["episode_state_restored"] is False
        for frames in session.collector._history._frames:
            assert torch.equal(frames, frames[0:1].expand_as(frames))
        with session:
            for _ in range(2):
                record = session.step()
                assert record["batch_samples"] == 12
                assert record["optimization"]["optimizer_steps"] > 0
            assert session.update == session.consumed_updates == (stage + 1) * 2
            assert session.collected_transitions == (stage + 1) * 24
            previous_model = deepcopy(session.model.state_dict())
            previous_adam = deepcopy(session.trainer.optimizer.state_dict())
            previous_rng = capture_rng()
            previous_sampler = session.sampler.state_dict()
            assert previous_adam["state"]
            previous_path = tmp_path / f"stage_{stage}.pt"
            previous_receipt = session.save(previous_path)
            assert_same(capture_rng(), previous_rng)
        # The saved optimizer is a real accumulating Adam state, not a new one.
        _, trainer, _, update, metadata, rng = load_frame_checkpoint(previous_path)
        assert_same(trainer.optimizer.state_dict(), previous_adam)
        assert_same(rng, previous_rng)
        assert update == (stage + 1) * 2
        assert metadata["continuation_segment"]["fresh_transitions"] == 24
        steps = {int(state["step"].item()) for state in previous_adam["state"].values()}
        assert min(steps) >= (stage + 1) * 2


@pytest.mark.parametrize("override,match", [
    ({"expected_initial_model_sha256": "0" * 64}, "initial model SHA mismatch"),
    ({"expected_initial_model_sha256": None}, "lowercase 64-hex"),
    ({"expected_initial_model_sha256": False}, "lowercase 64-hex"),
    ({"expected_initial_model_sha256": "A" * 64}, "lowercase 64-hex"),
    ({"training_seed": True}, "training_seed"),
    ({"training_seed": 2**32}, "training_seed"),
    ({"retention_seed": 71}, "independent"),
    ({"retention_seed": 801}, "independent"),
    ({"evaluation_seeds": [False]}, "independent"),
    ({"rollout_steps": True}, "rollout_steps"),
    ({"device": "cuda"}, "explicit index"),
])
def test_fresh_preflight_failure_preserves_rng_and_precedes_environment_or_cuda(monkeypatch, override, match):
    config = configuration()
    expected = _model_state_sha256(initial_model(config)[0])
    before = capture_rng()
    calls = []
    def forbidden(*_, **__):
        calls.append("unexpected startup")
        raise AssertionError("environment/CUDA/seeding before guard")
    monkeypatch.setattr("transformer_rl.frame_continuation._seed", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    options = {"device": "cuda:0", "expected_initial_model_sha256": expected, **override}
    with pytest.raises(ValueError, match=match):
        start(config, forbidden, **options)
    assert calls == []
    assert_same(capture_rng(), before)


def test_fresh_model_construction_error_restores_caller_rng(monkeypatch):
    config = configuration()
    before = capture_rng()
    def broken(_):
        torch.rand(11)
        raise RuntimeError("CPU construction failed")
    monkeypatch.setattr("transformer_rl.frame_continuation.FrameActorCritic", broken)
    calls = []
    def forbidden(*_, **__):
        calls.append(True)
    with pytest.raises(RuntimeError, match="CPU construction"):
        FrameContinuation.start(config, forbidden, "packed_env:make_env", rollout_steps=4,
            training_seed=71, retention_seed=901, expected_initial_model_sha256="0" * 64)
    assert calls == []
    assert_same(capture_rng(), before)


def test_mutated_fresh_config_rejected_before_model_factory_or_rng(monkeypatch):
    config = configuration()
    config.control["policy_dt_s"] = -1
    before = capture_rng()
    def forbidden(*_, **__):
        raise AssertionError("preflight did not revalidate config")
    monkeypatch.setattr("transformer_rl.frame_continuation.FrameActorCritic", forbidden)
    with pytest.raises(ValueError, match="policy_dt_s"):
        FrameContinuation.start(config, forbidden, "packed_env:make_env", rollout_steps=4,
            training_seed=71, retention_seed=901, expected_initial_model_sha256="0" * 64)
    assert_same(capture_rng(), before)


def test_environment_change_requires_explicit_transition_and_complete_continuation(tmp_path):
    config = configuration()
    path = tmp_path / "fresh.pt"
    with start(config) as session:
        receipt = session.save(path)
    changed = replace(config, environment={**config.environment, "stage": "next"})
    with pytest.raises(ValueError, match="exact continuation configuration"):
        resume(changed, path, receipt, update=0, transitions=0)
    ordinary, ordinary_path, ordinary_sha = parent(tmp_path)
    with pytest.raises(ValueError, match="unsupported format"):
        resume(ordinary, ordinary_path, {"sha256": ordinary_sha},
            update=2, transitions=24, environment_transition=True)


def test_exact_resume_after_transition_preserves_parent_binding_and_does_not_repeat_stage_claim(tmp_path):
    config = configuration()
    first = tmp_path / "initial.pt"
    with start(config) as session:
        first_receipt = session.save(first)
    changed = replace(config, environment={**config.environment, "stage": "next"})
    second = tmp_path / "transition.pt"
    with resume(changed, first, first_receipt, update=0, transitions=0,
            environment_transition=True) as session:
        session.step()
        second_receipt = session.save(second)
        assert session.metadata["stage_transition"] == {
            "environment_transition": True,
            "parent_environment_sha256": digest(config.environment),
            "environment_sha256": digest(changed.environment)}
        assert session.metadata["continuation_parent"] == {
            "path": str(first.resolve()), "sha256": first_receipt["sha256"],
            "update": 0, "resume": True}
    third = tmp_path / "exact_resume.pt"
    with resume(changed, second, second_receipt, update=1, transitions=12) as session:
        assert "stage_transition" not in session.metadata
        assert session.metadata["continuation_parent"] == {
            "path": str(second.resolve()), "sha256": second_receipt["sha256"],
            "update": 1, "resume": True}
        session.save(third)
    assert "stage_transition" not in load_frame_checkpoint(third)[4]
    assert load_frame_checkpoint(second)[4]["stage_transition"]["environment_transition"] is True


@pytest.mark.parametrize("flag,resume_mode", [(1, True), (None, True), ("yes", True), (True, False)])
def test_transition_flag_rejected_before_checkpoint_read(tmp_path, monkeypatch, flag, resume_mode):
    def forbidden(*_, **__):
        raise AssertionError("invalid flag read a checkpoint")
    monkeypatch.setattr("transformer_rl.frame_continuation._file_sha", forbidden)
    with pytest.raises(ValueError, match="environment_transition"):
        FrameContinuation.open(configuration(), forbidden, "packed_env:make_env", tmp_path / "absent.pt",
            checkpoint_sha256="0" * 64, parent_update=0, cumulative_transitions=0,
            consumed_updates=0, rollout_steps=4, training_seed=71, retention_seed=901,
            resume=resume_mode, environment_transition=flag)


@pytest.mark.parametrize("mutation,override,match", [
    ("ppo", {}, "PPO recipe"), ("model", {}, "PPO recipe"), ("control", {}, "PPO recipe"),
    (None, {"training_seed": 72}, "training seed"),
    (None, {"retention_seed": 902}, "seed"),
    (None, {"retention_coef": .1}, "anchor paths"),
    (None, {"retention_batch_size": 128}, "identity mismatch"),
    (None, {"evaluation_seeds": (803,)}, "evaluation seeds"),
    (None, {"rollout_steps": 5}, "clock"),
    (None, {"consumed_updates": 1}, "clock"),
    ("source", {}, "source and device"), ("device", {}, "source and device"),
])
def test_transition_never_relaxes_non_environment_identity(tmp_path, monkeypatch, mutation, override, match):
    config = configuration()
    path = tmp_path / "fresh.pt"
    with start(config) as session:
        receipt = session.save(path)
    changed = replace(config, environment={**config.environment, "stage": "next"})
    if mutation == "ppo":
        changed = replace(changed, ppo=replace(changed.ppo, learning_rate=2e-3))
    elif mutation == "model":
        changed = replace(changed, model=replace(changed.model, initial_std=.9))
    elif mutation == "control":
        changed = replace(changed, control={**changed.control, "target_offset": [0., 0.]})
    elif mutation in ("source", "device"):
        payload = torch.load(path, weights_only=True)
        payload["metadata"]["source" if mutation == "source" else "continuation_device"] = (
            {"files": {}, "sha256": "0" * 64} if mutation == "source" else "cuda:0")
        path = tmp_path / "tampered.pt"
        torch.save(payload, path)
        receipt = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    before = capture_rng()
    def forbidden(*_, **__):
        raise AssertionError("invalid identity constructed an environment or seeded")
    monkeypatch.setattr("transformer_rl.frame_continuation._seed", forbidden)
    with pytest.raises(ValueError, match=match):
        resume(changed, path, receipt, update=0, transitions=0, factory=forbidden,
            environment_transition=True, **override)
    assert_same(capture_rng(), before)


def test_fresh_partial_tail_is_charged_without_model_adam_or_private_rng_update(tmp_path):
    with start(configuration()) as session:
        model, adam, private = (deepcopy(session.model.state_dict()),
            deepcopy(session.trainer.optimizer.state_dict()), session.sampler.state_dict())
        ticks = iter([False, True])
        assert session.step(should_stop=lambda: next(ticks)) is None
        assert session.update == session.consumed_updates == 0
        assert session.collected_transitions == session.discarded_transitions == 3
        assert_same(session.model.state_dict(), model)
        assert_same(session.trainer.optimizer.state_dict(), adam)
        assert_same(session.sampler.state_dict(), private)
        path = tmp_path / "partial.pt"
        receipt = session.save(path)
    with resume(configuration(), path, receipt, update=0, transitions=3) as recovered:
        assert recovered.collected_transitions == 3
        assert recovered.consumed_updates == 0
        assert_same(recovered.model.state_dict(), model)
        assert_same(recovered.trainer.optimizer.state_dict(), adam)


def test_transition_next_ppo_update_matches_manual_complete_state_restore(tmp_path):
    config = configuration()
    path = tmp_path / "stage_a.pt"
    with start(config) as session:
        session.step()
        session.step()
        receipt = session.save(path)
    changed = replace(config, environment={**config.environment, "stage": "b"})
    with resume(changed, path, receipt, update=2, transitions=24,
            factory=noisy_factory([]), environment_transition=True) as continued:
        continued.step()
        actual_model = deepcopy(continued.model.state_dict())
        actual_adam = deepcopy(continued.trainer.optimizer.state_dict())
        actual_rng = capture_rng()
    model, trainer, _, _, _, saved_rng = load_frame_checkpoint(path)
    _seed(71)
    env = noisy_factory([])(model_config=changed.model,
        environment_config=changed.environment, device=torch.device("cpu"))
    collector = FrameCollector(env, model, changed.ppo, changed.control["action_bounds"])
    collector.reset(seed=71)
    restore_rng(saved_rng)
    trainer.update(collector.collect(4), diagnostics=True)
    assert_same(model.state_dict(), actual_model)
    assert_same(trainer.optimizer.state_dict(), actual_adam)
    assert_same(capture_rng(), actual_rng)
    env.close()


def test_transition_preserves_positive_retention_draws_and_rejects_lambda_batch_or_pool_changes(tmp_path):
    config, ordinary_path, ordinary_sha = parent(tmp_path)
    actor = load_frame_checkpoint(ordinary_path)[0].actor
    frames = torch.randn(7, 4, 5, generator=torch.Generator().manual_seed(312))
    mean = actor.policy(frames).detach() + .2
    anchor = tmp_path / "anchors.pt"
    save_anchors(anchor, config, frames, mean, actor.log_std.exp().expand_as(mean), ordinary_sha)
    path = tmp_path / "retained.pt"
    with FrameContinuation.open(config, make_env, "packed_env:make_env", ordinary_path,
            checkpoint_sha256=ordinary_sha, parent_update=2, cumulative_transitions=24,
            consumed_updates=2, rollout_steps=4, training_seed=71, retention_seed=901,
            evaluation_seeds=(801, 802), anchors=[anchor], retention_coef=.3,
            retention_batch_size=3) as session:
        session.step()
        receipt = session.save(path)
        state = session.sampler.state_dict()
        assert state["draw_count"] > 0 and state["pool_sizes"] == [7]
        next_loss = session.sampler().detach()
        next_state = session.sampler.state_dict()
    changed = replace(config, environment={**config.environment, "stage": "b"})
    options = dict(environment_transition=True, anchors=[anchor], retention_coef=.3,
        retention_batch_size=3)
    with resume(changed, path, receipt, update=3, transitions=36, **options) as continued:
        assert_same(continued.sampler.state_dict(), state)
        rng = capture_rng()
        assert_same(continued.sampler().detach(), next_loss)
        assert_same(continued.sampler.state_dict(), next_state)
        assert_same(capture_rng(), rng)
    other = tmp_path / "other_anchors.pt"
    save_anchors(other, config, frames[:4], mean[:4],
        actor.log_std.exp().expand_as(mean[:4]), ordinary_sha)
    for override in ({"retention_coef": .2}, {"retention_batch_size": 2},
                     {"retention_seed": 902}, {"anchors": [other]}):
        with pytest.raises(ValueError, match="identity mismatch"):
            resume(changed, path, receipt, update=3, transitions=36, **{**options, **override})


def test_failed_ppo_with_actual_parameter_mutation_invalidates_fresh_session(tmp_path, monkeypatch):
    with start(configuration()) as session:
        def failing(*_, **__):
            with torch.no_grad():
                next(session.model.parameters()).add_(1)
            raise FloatingPointError("failed after mutation")
        monkeypatch.setattr(session.trainer, "update", failing)
        with pytest.raises(FloatingPointError, match="mutation"):
            session.step()
        assert session.invalid and session.update == 0 and session.consumed_updates == 1
        assert session.collected_transitions == 12
        with pytest.raises(RuntimeError, match="invalid"):
            session.save(tmp_path / "unsafe.pt")
        with pytest.raises(RuntimeError, match="invalid"):
            session.step()
        assert not (tmp_path / "unsafe.pt").exists()


@pytest.mark.parametrize("entry", ["start", "transition"])
@pytest.mark.parametrize("failure", ["factory", "reset", "provenance"])
def test_startup_failure_invalidates_session_even_if_close_raises(tmp_path, entry, failure):
    config = configuration()
    path = tmp_path / "parent.pt"
    with start(config) as original:
        receipt = original.save(path)
    captured = []
    class Observed(FrameContinuation):
        def close(self):
            captured.append(self)
            super().close()
    def failing(**kwargs):
        if failure == "factory":
            raise ValueError("original factory failure")
        env = make_env(**kwargs)
        if failure == "reset":
            def reset(*_, **__):
                raise ValueError("original reset failure")
            env.reset = reset
        else:
            env.metadata["identity"] = "different source"
            if entry == "start":
                env.metadata["control_sha256"] = "0" * 64
        def close():
            raise RuntimeError("secondary shutdown failure")
        env.close = close
        return env
    with pytest.raises(ValueError) as caught:
        if entry == "start":
            Observed.start(config, failing, "packed_env:make_env", rollout_steps=4,
                training_seed=71, retention_seed=901,
                expected_initial_model_sha256=_model_state_sha256(initial_model(config)[0]))
        else:
            Observed.open(config, failing, "packed_env:make_env", path,
                checkpoint_sha256=receipt["sha256"], parent_update=0,
                cumulative_transitions=0, consumed_updates=0, rollout_steps=4,
                training_seed=71, retention_seed=901, evaluation_seeds=(801, 802),
                resume=True, environment_transition=True)
    assert len(captured) == 1
    assert captured[0].closed and captured[0].invalid
    for operation in (captured[0].step, lambda: captured[0].save(tmp_path / "unsafe.pt")):
        with pytest.raises(RuntimeError, match="invalid"):
            operation()
    if failure != "factory":
        assert any("secondary shutdown failure" in note for note in caught.value.__notes__)

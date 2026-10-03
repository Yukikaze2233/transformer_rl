"""CPU interface checks for fixed transfer masks and actual domain readback."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from transformer_rl.transfer_profiles import (
    MODULES, REPRESENTATIVE_CASES, apply_reset_profiles, build_transfer_cases,
    configure_transfer_evaluation,
)


def base_cases():
    return [{"name": name, "reset_seed_key": name, "terrain": "flat", "task": "survive",
             "skill": {"kind": name, "command": [0., 0., .305]}, "terrain_seed": 91,
             "episode_seconds": 10.} for name in REPRESENTATIVE_CASES]


def base_config():
    result = {"physics_dt": .005, "policy_dt": .01, "evaluation_exact_cases": True,
              "evaluation": {"cases": base_cases(), "frozen_module_enablement": {"signal_delay": False},
                             "frozen_signal_perturbations": {"enabled": False}, "stable_case_layout": True},
              "usb_transport": {"enabled": False}}
    result.update({name: {"enabled": False} for name in MODULES})
    result["signal_delay"].update(version="physical_signal_delay_v1", feedback_pd="fresh_physics_feedback",
                                  schedule={"start": 1500, "end": 2500}, enabled_fraction=.75, jitter_ms=5.)
    return result


class Delay:
    """Exercise the baseline's schedule/jitter contract without a simulator."""
    def __init__(self, count, config):
        self.cfg, self.dt_ms = deepcopy(config), 5.
        self.enabled = torch.zeros(count, dtype=torch.bool)
        self.profile = torch.zeros(count, dtype=torch.long)
        self.base_ms = torch.zeros(count, 6)
        self.bounds_ms = torch.zeros(count, 6, 2)
        self.lags = torch.zeros(count, 6, dtype=torch.long)
        self.strength = 0.
        self.calls = []

    def begin_policy(self, actor_update):
        self.calls.append(actor_update)
        schedule = self.cfg["schedule"]
        self.strength = min(1., max(0., (actor_update - schedule["start"]) / (schedule["end"] - schedule["start"])))
        ms = (self.base_ms + 5.).clamp(min=self.bounds_ms[..., 0], max=self.bounds_ms[..., 1])
        self.lags.copy_((ms * self.strength / self.dt_ms).round().long() * self.enabled[:, None])


class Noise:
    def __init__(self, count):
        self.frame_dim, self.max_lag = 35, 0
        self.sensor_indices = list(range(4, 22))
        self.noise = torch.zeros(35)
        self.noise[4:7], self.noise[7:10], self.noise[10:14], self.noise[16:22] = .0025, .005, .003, .005
        self.enabled = torch.zeros(count, dtype=torch.bool)
        self.obs_lag = torch.zeros(count, dtype=torch.long)
        self.act_lag = torch.zeros(count, dtype=torch.long)
        self.cached = torch.ones(count, dtype=torch.bool)
        self.observations = SimpleNamespace(reset=self.reset_observations)

    def reset_observations(self, rows):
        self.cached[rows] = False

    def set_enabled(self, rows, enabled):
        self.enabled[rows] = enabled
        self.obs_lag[rows] *= enabled
        self.act_lag[rows] *= enabled


def environment():
    cases = build_transfer_cases(base_cases())
    config = configure_transfer_evaluation(base_config(), cases)
    count = len(cases)
    dynamics = SimpleNamespace(base_id=0, wheel_ids=[1, 2], mass_scale=torch.ones(count, 4),
                               inertia_scale=torch.ones(count, 4), com_offset=torch.zeros(count, 3),
                               enabled=torch.zeros(count, dtype=torch.bool))
    contact = SimpleNamespace(mu=torch.full((count,), .5))
    transport = SimpleNamespace(enabled=torch.zeros(count, dtype=torch.bool), low=.00004,
                                base_delay=torch.full((count,), .00004, dtype=torch.float64),
                                loss_probability=torch.zeros(count), phase_aware=True,
                                can_phase=torch.full((count,), .0005, dtype=torch.float64),
                                jitter_half_width=torch.zeros(count, dtype=torch.float64),
                                phase_low=0., phase_high=.001)
    for row, case in enumerate(cases):
        profile = case["dynamics_profile"]
        if profile:
            dynamics.enabled[row] = True
            dynamics.mass_scale[row, 0] = profile["base_mass_scale"]
            dynamics.com_offset[row] = torch.tensor(profile["base_com_offset_m"])
        if case["contact_profile"]:
            contact.mu[row] = case["contact_profile"]["friction"]
        if case["communication_profile"]:
            transport.enabled[row] = True
            transport.base_delay[row] = case["communication_profile"]["delay_ms"] * .001
            transport.can_phase[row] = case["communication_profile"]["can_phase_ms"] * .001
    dynamics.masses = dynamics.mass_scale * torch.tensor([4., 1., 1., 2.])
    surface_mu = torch.stack((contact.mu, contact.mu, torch.zeros(count)), -1)
    return SimpleNamespace(cfg=config, num_envs=count, scene_groups=[case["name"] for case in cases],
                           clone_indices=list(range(count)), surface_mu=surface_mu,
                           surface_valid=torch.tensor([[True, True, False]] * count),
                           dt=.005, device="cpu", signal_delay=Delay(count, config["signal_delay"]),
                           perturbations=Noise(count), dynamics_randomization=dynamics, contact_domain=contact,
                           command_transport=transport, usb_transport=None, body_mass=dynamics.masses.clone(),
                           motor_strength=torch.full((count, 6), .87), spring_strength=torch.full((count, 1), 1.08),
                           skills=SimpleNamespace(schedule=SimpleNamespace(motor_scale=torch.full((count, 6), .5))))


def test_case_recipes_are_paired_and_cover_all_declared_factors_without_mutating_input():
    source = base_cases()
    before = deepcopy(source)
    cases = build_transfer_cases(source)
    assert source == before
    assert len(cases) == len({case["name"] for case in cases}) == 50
    assert {case["transfer_profile"]["base_case"] for case in cases} == set(REPRESENTATIVE_CASES)
    for base in REPRESENTATIVE_CASES:
        paired = [case for case in cases if case["transfer_profile"]["base_case"] == base]
        assert len(paired) == (10 if base == "stand_305mm" else 8)
        assert all(case["reset_seed_key"] == base and case["terrain_seed"] == 91 for case in paired)
    combined = cases[7]
    assert combined["transfer_profile"]["signal_age_ms"] == [40.] * 4 + [30.] * 2
    assert combined["transfer_profile"]["noise_enabled"]
    assert combined["dynamics_profile"]["base_com_offset_m"] == [.01, .005, 0.]
    assert combined["communication_profile"]["jitter_ms"] == 0.
    assert cases[-2]["transfer_profile"]["motor_strength"] == .85
    assert cases[-1]["transfer_profile"]["spring_strength"] == .9


def test_case_recipe_rejects_missing_or_duplicate_task_definitions():
    with pytest.raises(ValueError, match="six representative"):
        build_transfer_cases(base_cases()[:-1])
    with pytest.raises(ValueError, match="unique"):
        build_transfer_cases(base_cases() + base_cases()[:1])


def test_config_reverses_nominal_materializer_overrides_explicitly_and_keeps_training_unchanged():
    source = base_config()
    source["evaluation_exact_cases"] = False
    source["new_asset_curriculum"] = {"enabled": True, "request_protocol": {"new_probability": 1.}}
    result = configure_transfer_evaluation(source, build_transfer_cases(base_cases()))
    assert source["signal_delay"]["enabled"] is False
    assert source["evaluation_exact_cases"] is False and source["new_asset_curriculum"]["enabled"]
    assert result["evaluation_exact_cases"] and result["new_asset_curriculum"]["enabled"]
    assert set(result["skill_specs"]) == {case["name"] for case in result["evaluation"]["cases"]}
    assert all(result[name]["enabled"] is True for name in MODULES)
    assert result["signal_perturbations"]["max_delay_steps"] == 0
    assert result["signal_perturbations"]["noise_scale"] == 1.
    assert result["observation_noise_enabled"]
    assert "frozen_signal_perturbations" not in result["evaluation"]
    assert result["evaluation"]["frozen_module_enablement"]["signal_delay"]
    assert result["signal_delay"]["enabled_fraction"] == .75
    assert result["signal_delay"]["jitter_ms"] == 5.


@pytest.mark.parametrize("change", ("physics", "usb", "missing_module", "invalid_age", "unknown_profile"))
def test_config_rejects_unsupported_or_ambiguous_contracts(change):
    config, cases = base_config(), build_transfer_cases(base_cases())
    if change == "physics":
        config["physics_dt"] = .001
    elif change == "usb":
        config["usb_transport"]["enabled"] = True
    elif change == "missing_module":
        del config["signal_delay"]
    elif change == "invalid_age":
        cases[0]["transfer_profile"]["signal_age_ms"][0] = 7.
    else:
        cases[0]["transfer_profile"]["invented_sensor"] = 1
    with pytest.raises(ValueError):
        configure_transfer_evaluation(config, cases)


def test_runtime_configuration_applies_every_case_and_records_actual_readback():
    env = environment()
    metadata = configure_transfer_evaluation(env)
    assert len(metadata["verified_rows"]) == 50
    assert metadata["delay_curriculum_strength"] == 1.
    assert env.task_actor_update == env.stage_actor_update == 2500
    assert env.signal_delay.lags[3].tolist() == [16] * 4 + [12] * 2
    assert env.signal_delay.lags[0].tolist() == [0] * 6
    assert env.perturbations.enabled.nonzero().flatten().tolist() == [i for i in range(48) if i % 8 in (4, 7)]
    assert not env.perturbations.cached.any()
    assert env.skills.schedule.motor_scale.eq(1).all()
    assert metadata["verified_rows"][7]["downlink_enabled"]
    assert metadata["verified_rows"][0]["downlink_delay_ms"] == 0.
    assert metadata["verified_rows"][-2]["motor_strength"] == pytest.approx([.85] * 6)
    assert metadata["verified_rows"][-1]["spring_strength"] == pytest.approx(.9)


def test_delay_is_full_strength_on_later_zero_update_policy_calls_and_wrapper_is_installed_once():
    env = environment()
    configure_transfer_evaluation(env)
    begin = env.signal_delay.begin_policy
    before = env.signal_delay.lags.clone()
    env.signal_delay.begin_policy(0)
    env.signal_delay.begin_policy(1501)
    torch.testing.assert_close(env.signal_delay.lags, before, atol=0, rtol=0)
    apply_reset_profiles(env, torch.tensor([3]))
    assert env.signal_delay.begin_policy is begin
    assert set(env.signal_delay.calls) == {2500}
    assert env.cfg["signal_delay"]["schedule"] == {"start": 1500, "end": 2500}


def test_contact_readback_tracks_solver_rows_through_nonidentity_clone_order():
    env = environment()
    # Move low-grip and combined scenes onto originally nominal solver rows.
    permutation = list(range(env.num_envs))
    permutation[0], permutation[6] = permutation[6], permutation[0]
    permutation[1], permutation[7] = permutation[7], permutation[1]
    env.clone_indices = permutation
    env.scene_groups = [env.scene_groups[index] for index in permutation]
    for name in ("mass_scale", "inertia_scale", "com_offset", "enabled", "masses"):
        setattr(env.dynamics_randomization, name, getattr(env.dynamics_randomization, name)[permutation])
    for name in ("enabled", "base_delay", "loss_probability", "can_phase", "jitter_half_width"):
        setattr(env.command_transport, name, getattr(env.command_transport, name)[permutation])
    env.body_mass = env.body_mass[permutation]
    env.surface_mu = env.surface_mu[permutation]
    env.surface_valid = env.surface_valid[permutation]
    assert env.contact_domain.mu[0] == .5 and env.surface_mu[0, 0] == pytest.approx(.3)
    metadata = configure_transfer_evaluation(env)
    assert metadata["verified_rows"][0]["case"] == "stand_305mm__low_grip"
    assert metadata["verified_rows"][0]["friction"] == pytest.approx(.3)
    assert metadata["verified_rows"][6]["case"] == "stand_305mm__nominal"
    assert metadata["verified_rows"][6]["friction"] == pytest.approx(.5)
    env.surface_mu[0, 1] = .5
    with pytest.raises(ValueError, match="surface_friction"):
        apply_reset_profiles(env, torch.tensor([0]))


def test_invalid_clone_permutation_rejects_contact_readback():
    env = environment()
    env.clone_indices[0] = env.clone_indices[1]
    with pytest.raises(ValueError, match="clone permutation"):
        configure_transfer_evaluation(env)


def test_partial_reset_restores_only_selected_masks_strengths_and_noise_cache():
    env = environment()
    configure_transfer_evaluation(env)
    env.signal_delay.base_ms[3] = 2.
    env.signal_delay.bounds_ms[3] = 3.
    env.signal_delay.enabled[3] = False
    env.motor_strength[3] = .86
    env.spring_strength[3] = 1.09
    env.perturbations.enabled[3:5] = True
    env.perturbations.obs_lag[3] = env.perturbations.act_lag[3] = 2
    env.perturbations.cached[:] = True
    env.motor_strength[4] = .91
    metadata = apply_reset_profiles(env, torch.tensor([3]))
    assert [record["row"] for record in metadata["verified_rows"]] == [3]
    assert env.signal_delay.base_ms[3].tolist() == [80.] * 4 + [60.] * 2
    assert not env.perturbations.enabled[3] and env.perturbations.enabled[4]
    assert env.perturbations.obs_lag[3] == env.perturbations.act_lag[3] == 0
    assert not env.perturbations.cached[3] and env.perturbations.cached[4]
    assert env.motor_strength[3].eq(1).all() and env.motor_strength[4].eq(.91).all()


@pytest.mark.parametrize("fault,match", (
    ("mass", "physical_mass"), ("com", "com_offset"), ("contact", "friction"),
    ("enabled", "downlink_enabled"), ("delay", "downlink_base"), ("phase", "can_phase"),
    ("noise", "sensor_noise_std"), ("noise_lag", "without policy delay"),
    ("clock", "wrong physical quantum"), ("missing", "no runtime object"),
))
def test_runtime_requires_effective_modules_and_checks_realized_domains(fault, match):
    env = environment()
    if fault == "mass":
        env.body_mass[0, 0] += .1
    elif fault == "com":
        env.dynamics_randomization.com_offset[5, 0] = 0.
    elif fault == "contact":
        env.contact_domain.mu[6] = .5
    elif fault == "enabled":
        env.command_transport.enabled[0] = True
    elif fault == "delay":
        env.command_transport.base_delay[7] = .00008
    elif fault == "phase":
        env.command_transport.can_phase[7] = .0005
    elif fault == "noise":
        env.perturbations.noise[4] *= 2
    elif fault == "noise_lag":
        env.perturbations.max_lag = 1
    elif fault == "clock":
        env.signal_delay.dt_ms = 1.
    else:
        env.signal_delay = None
    with pytest.raises(ValueError, match=match):
        configure_transfer_evaluation(env)


def test_undeclared_legacy_evaluation_and_training_are_no_ops():
    for config in ({}, {"evaluation_exact_cases": True}, {"evaluation_exact_cases": False}):
        assert apply_reset_profiles(SimpleNamespace(cfg=config), [0]) is None
    env = environment()
    env.cfg["evaluation_exact_cases"] = False
    with pytest.raises(ValueError, match="explicit exact-evaluation"):
        configure_transfer_evaluation(env)


@pytest.mark.parametrize("rows", ([0, 0], [-1], [50], [[0]], [0.5]))
def test_invalid_reset_indices_are_rejected(rows):
    with pytest.raises(ValueError, match="Reset rows"):
        apply_reset_profiles(environment(), rows)

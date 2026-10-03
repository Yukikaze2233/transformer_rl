"""Fixed, explicitly verified transfer probes for a frozen V6 environment.

The baseline owns physical buffers, body properties and transport simulation.
This module selects existing fixed-case domains and replaces reset-time random
sensor/effort masks only in an explicitly declared evaluation. These profiles
are engineering sensitivity tests, not identified hardware uncertainty.
"""
from __future__ import annotations

from copy import deepcopy
import math


VERSION = "fixed_transfer_profiles_v1"
MODULES = ("signal_delay", "signal_perturbations", "dynamics_randomization",
           "contact_domain", "command_transport")
REPRESENTATIVE_CASES = ("stand_305mm", "forward_05", "backward_05", "rotate_1",
                        "start_stop_05", "height_scan")
AGE_GROUPS = ("joint_position", "joint_velocity", "gyro", "gravity", "leg_target", "wheel_target")


def _number(value, low, high, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and within [{low}, {high}]")
    return float(value)


def _profile(case):
    profile = case.get("transfer_profile")
    fields = {"name", "base_case", "signal_age_ms", "noise_enabled", "motor_strength", "spring_strength"}
    if not isinstance(profile, dict) or set(profile) != fields:
        raise ValueError("Every transfer case requires the complete fixed profile schema")
    for key in ("name", "base_case"):
        if not isinstance(profile[key], str) or not profile[key]:
            raise ValueError(f"Transfer {key} must be a nonempty string")
    ages = profile["signal_age_ms"]
    if not isinstance(ages, list) or len(ages) != 6:
        raise ValueError("Transfer ages require four sensor and two target groups")
    for index, age in enumerate(ages):
        value = _number(age, 0., 80. if index < 4 else 60., AGE_GROUPS[index])
        if not math.isclose(value / 5., round(value / 5.), abs_tol=1e-9):
            raise ValueError("Fixed ages must be multiples of the 5 ms physical quantum")
    if type(profile["noise_enabled"]) is not bool:
        raise ValueError("Transfer noise_enabled must be explicit")
    _number(profile["motor_strength"], .85, 1., "motor_strength")
    _number(profile["spring_strength"], .9, 1.1, "spring_strength")
    return profile


def build_transfer_cases(basecases):
    """Build 50 paired probes from six existing fixed task definitions.

    Each base task is tested in eight common profiles. Two extra standing
    probes isolate motor and spring scaling, which also appear in the combined
    profile. Nominal reset_seed_key stays paired with its source task.
    """
    mapping = {case["name"]: case for case in basecases}
    if len(mapping) != len(basecases) or set(REPRESENTATIVE_CASES) - set(mapping):
        raise ValueError("Transfer probes require unique definitions for all six representative tasks")
    recipes = (("nominal", 0., 0., False), ("delay20_15", 20., 15., False),
               ("delay40_30", 40., 30., False), ("delay80_60", 80., 60., False),
               ("noise", 0., 0., True), ("payload_com", 0., 0., False),
               ("low_grip", 0., 0., False), ("combined", 40., 30., True))
    result = []

    def append(base_name, name, observation=0., action=0., noise=False):
        case = deepcopy(mapping[base_name])
        case.update(name=f"{base_name}__{name}", reset_seed_key=case.get("reset_seed_key", base_name),
                    dynamics_profile=None, contact_profile=None, communication_profile=None)
        case["transfer_profile"] = {"name": name, "base_case": base_name,
            "signal_age_ms": [observation] * 4 + [action] * 2, "noise_enabled": noise,
            "motor_strength": .85 if name in ("motor_weak", "combined") else 1.,
            "spring_strength": .9 if name in ("spring_weak", "combined") else 1.}
        if name in ("payload_com", "combined"):
            case["dynamics_profile"] = {"base_mass_scale": 1.1, "base_com_offset_m": [.01, .005, 0.]}
        if name in ("low_grip", "combined"):
            case["contact_profile"] = {"friction": .3}
        if name == "combined":
            case["communication_profile"] = {"delay_ms": .16, "can_phase_ms": 1., "jitter_ms": 0.,
                                             "drop_probability": 0.}
        _profile(case)
        result.append(case)

    for base in REPRESENTATIVE_CASES:
        for name, observation, action, noise in recipes:
            append(base, name, observation, action, noise)
    append("stand_305mm", "motor_weak")
    append("stand_305mm", "spring_weak")
    return result


def configure_transfer_evaluation(config, cases=None):
    """Prepare a config, or configure/read back an already constructed env.

    Config input may be the materialized training contract. The returned copy
    materializes fixed task semantics and reverses clean-evaluation delay/noise
    overrides explicitly. Mixed profiles may share one simulation batch.
    """
    if not isinstance(config, dict):
        if cases is not None:
            raise ValueError("Runtime transfer configuration does not accept replacement cases")
        import torch
        return apply_reset_profiles(config, torch.arange(config.num_envs, device=config.motor_strength.device))
    result = deepcopy(config)
    if result.get("physics_dt") != .005:
        raise ValueError("Transfer evaluation requires the V6 5 ms physical clock")
    if result.get("usb_transport", {}).get("enabled"):
        raise ValueError("Transfer probes cannot double-count the legacy USB RTT layer")
    suite = result["evaluation"]
    if cases is not None:
        suite["cases"] = deepcopy(cases)
    selected = suite["cases"]
    if not selected or len({case["name"] for case in selected}) != len(selected):
        raise ValueError("Transfer cases must be nonempty and uniquely named")
    for case in selected:
        _profile(case)
    result.update(evaluation_exact_cases=True, record_diagnostics=True, diagnostic_logging=False,
                  evaluation_long_corridors=bool(result.get("task_semantics")))
    if "episode_seconds" in suite:
        result["episode_seconds"] = suite["episode_seconds"]
    if "fall_confirmation_seconds" in result:
        result["fall_confirmation_seconds"] = 0.
    result.pop("command_curriculum", None)
    result["skill_specs"] = {case["name"]: deepcopy(case.get("skill", {})) for case in selected}
    # Exact cases already replace sampling. Keep the curriculum interface active
    # for the frozen runtime's dense tracking and command-reference contract.
    for assistance in ("jump_assist", "step_assist"):
        if assistance in result:
            result[assistance]["enabled"] = False
    for module in MODULES:
        if not isinstance(result.get(module), dict):
            raise ValueError(f"Frozen V6 contract does not provide {module}")
        result[module]["enabled"] = True
    signal = result["signal_delay"]
    if signal.get("version") != "physical_signal_delay_v1" or signal.get("feedback_pd") != "fresh_physics_feedback":
        raise ValueError("Transfer probes require the audited physical sensor/setpoint delay module")
    end = signal.get("schedule", {}).get("end")
    if type(end) is not int or end < 1:
        raise ValueError("Transfer evaluation requires a valid delay curriculum endpoint")
    result["signal_perturbations"].update(max_delay_steps=0, noise_scale=1., enabled_fraction=1.)
    result["observation_noise_enabled"] = True
    suite["frozen_module_enablement"] = {**suite.get("frozen_module_enablement", {}),
                                        **{name: True for name in MODULES}}
    suite.pop("frozen_signal_perturbations", None)
    suite["stable_case_layout"] = False
    result["scene_groups"] = [{"name": case["name"], "fraction": 1 / len(selected),
                                "terrain": [case.get("terrain", "flat")]} for case in selected]
    result["transfer_evaluation"] = {"version": VERSION, "physics_dt": .005,
        "modules": list(MODULES), "evaluation_actor_update": end,
        "scope": "fixed_engineering_sensitivity_profiles_not_hardware_identified_uncertainty"}
    return result


def _context(env):
    declaration = env.cfg.get("transfer_evaluation")
    if declaration is None:
        return None
    if (env.cfg.get("evaluation_exact_cases") is not True or not isinstance(declaration, dict)
            or declaration.get("version") != VERSION or declaration.get("physics_dt") != .005
            or getattr(env, "dt", None) != .005 or declaration.get("modules") != list(MODULES)):
        raise ValueError("Fixed transfer profiles require an explicit exact-evaluation declaration")
    for name in MODULES:
        if env.cfg.get(name, {}).get("enabled") is not True:
            raise ValueError(f"Transfer evaluation requires instantiated {name}")
    if getattr(env, "usb_transport", None) is not None:
        raise ValueError("Transfer evaluation has an unexpected legacy USB RTT layer")
    attrs = {"signal_delay": "signal_delay", "signal_perturbations": "perturbations",
             "dynamics_randomization": "dynamics_randomization", "contact_domain": "contact_domain",
             "command_transport": "command_transport"}
    for name, attr in attrs.items():
        if getattr(env, attr, None) is None:
            raise ValueError(f"Enabled {name} has no runtime object")
    cases = env.cfg["evaluation"]["cases"]
    mapping = {case["name"]: case for case in cases}
    if len(mapping) != len(cases) or len(env.scene_groups) != env.num_envs:
        raise ValueError("Transfer case layout is inconsistent")
    if set(env.scene_groups) != set(mapping):
        raise ValueError("Every transfer case must have actual evaluation rows")
    for case in cases:
        _profile(case)
    return declaration, [mapping[name] for name in env.scene_groups]


def _rows(env, rows):
    import torch
    selected = torch.as_tensor(rows, device=env.motor_strength.device)
    if selected.ndim != 1 or selected.dtype not in (torch.int32, torch.int64):
        raise ValueError("Reset rows must be a one-dimensional integer index tensor")
    selected = selected.long()
    if (selected.numel() != selected.unique().numel() or bool((selected < 0).any())
            or bool((selected >= env.num_envs).any())):
        raise ValueError("Reset rows must be unique and within the evaluation layout")
    return selected


def _close(actual, expected, name, *, atol=1e-6):
    import torch
    expected = torch.as_tensor(expected, device=actual.device, dtype=actual.dtype)
    if actual.shape != expected.shape or not torch.allclose(actual, expected, atol=atol, rtol=0.):
        raise ValueError(f"Transfer runtime readback differs: {name}")


def _noise_vector(perturbations):
    import torch
    if perturbations.frame_dim != 35 or perturbations.max_lag != 0:
        raise ValueError("Transfer noise requires the 35D sensor-only perturbation module without policy delay")
    noise = torch.zeros_like(perturbations.noise)
    noise[4:7], noise[7:10], noise[10:14], noise[16:22] = .0025, .005, .003, .005
    if list(perturbations.sensor_indices) != list(range(4, 22)):
        raise ValueError("Transfer noise sensor indices differ from the frozen 35D contract")
    _close(perturbations.noise, noise, "sensor_noise_std")
    return noise


def _verify_static(env, cases, selected):
    """Verify domains applied by the baseline constructor, including nominal rows."""
    import torch
    dynamics, contact, transport = env.dynamics_randomization, env.contact_domain, env.command_transport
    for row in selected.tolist():
        case = cases[row]
        profile = case.get("dynamics_profile") or {}
        expected = torch.full_like(dynamics.mass_scale[row], float(profile.get("leg_mass_scale", 1.)))
        expected[dynamics.base_id] = profile.get("base_mass_scale", 1.)
        expected[dynamics.wheel_ids] = profile.get("wheel_mass_scale", 1.)
        _close(dynamics.mass_scale[row], expected, f"{case['name']}.mass_scale")
        _close(dynamics.enabled[row], bool(profile), f"{case['name']}.dynamics_enabled", atol=0.)
        _close(dynamics.inertia_scale[row], torch.full_like(expected, profile.get("inertia_scale", 1.)),
               f"{case['name']}.inertia_scale")
        _close(dynamics.com_offset[row], profile.get("base_com_offset_m", [0., 0., 0.]),
               f"{case['name']}.com_offset")
        _close(env.body_mass[row], dynamics.masses[row], f"{case['name']}.physical_mass")
        _close(contact.mu[row], (case.get("contact_profile") or {}).get("friction", .5),
               f"{case['name']}.friction")
        communication = case.get("communication_profile")
        _close(transport.enabled[row], communication is not None, f"{case['name']}.downlink_enabled", atol=0.)
        expected_delay = transport.low if communication is None else communication["delay_ms"] * .001
        _close(transport.base_delay[row], expected_delay, f"{case['name']}.downlink_base_s", atol=1e-9)
        _close(transport.loss_probability[row], (communication or {}).get("drop_probability", 0.),
               f"{case['name']}.downlink_loss")
        if getattr(transport, "phase_aware", False):
            expected_phase = (communication or {}).get("can_phase_ms", .5 * (transport.phase_low + transport.phase_high) * 1000.)
            _close(transport.can_phase[row], expected_phase * .001, f"{case['name']}.can_phase_s", atol=1e-9)
            _close(transport.jitter_half_width[row], (communication or {}).get("jitter_ms", 0.) * .001,
                   f"{case['name']}.downlink_jitter_s", atol=1e-9)


def apply_reset_profiles(env, rows):
    """Apply and read back fixed probe masks after every explicit env.reset.

    No-op for training and legacy evaluation contracts without the declaration.
    The evaluation-only begin_policy wrapper always invokes the original delay
    implementation at full curriculum strength. Equal age bounds remove its
    random jitter without changing the baseline source or physical buffers.
    """
    context = _context(env)
    if context is None:
        return None
    import torch
    declaration, cases = context
    selected = _rows(env, rows)
    profiles = [_profile(case) for case in cases]
    delay, noise = env.signal_delay, env.perturbations
    if not math.isclose(float(delay.dt_ms), 5.):
        raise ValueError("Runtime signal delay has the wrong physical quantum")
    expected_noise = _noise_vector(noise)
    ages = delay.base_ms.new_tensor([profiles[row]["signal_age_ms"] for row in selected.tolist()]).reshape(-1, 6)
    delay.enabled[selected] = ages.gt(0).any(-1)
    delay.base_ms[selected] = ages
    delay.bounds_ms[selected] = ages[..., None].expand(-1, -1, 2)
    delay.profile[selected] = 0
    noise.set_enabled(selected, torch.tensor([profiles[row]["noise_enabled"] for row in selected.tolist()],
                                             dtype=torch.bool, device=noise.enabled.device))
    noise.obs_lag[selected], noise.act_lag[selected] = 0, 0
    # Constructor observation reads may have cached a randomly enabled sample
    # at this same policy tick. Invalidate only the reset rows so the next read
    # uses the exact noise mask, without reusing the previous episode.
    noise.observations.reset(selected)
    env.motor_strength[selected] = env.motor_strength.new_tensor(
        [profiles[row]["motor_strength"] for row in selected.tolist()])[:, None]
    env.spring_strength[selected] = env.spring_strength.new_tensor(
        [profiles[row]["spring_strength"] for row in selected.tolist()])[:, None]
    if getattr(getattr(env, "skills", None), "schedule", None) is not None:
        env.skills.schedule.motor_scale[selected] = 1.
    end = declaration["evaluation_actor_update"]
    if end != delay.cfg["schedule"]["end"]:
        raise ValueError("Declared full-strength clock differs from the delay module")
    env.task_actor_update = env.stage_actor_update = end
    if not getattr(delay, "_transfer_fixed_evaluation", False):
        original = delay.begin_policy

        def begin_fixed_policy(actor_update):
            original(end)

        delay.begin_policy = begin_fixed_policy
        delay._transfer_fixed_evaluation = True
    delay.begin_policy(0)
    _close(delay.base_ms[selected], ages, "sensor_and_target_base_ms")
    _close(delay.enabled[selected], ages.gt(0).any(-1), "signal_delay_enabled", atol=0.)
    _close(delay.bounds_ms[selected], ages[..., None].expand(-1, -1, 2), "sensor_and_target_bounds_ms")
    _close(delay.lags[selected], (ages / 5.).round().long(), "sensor_and_target_lags", atol=0.)
    if delay.strength != 1.:
        raise ValueError("Transfer evaluation delay curriculum strength was not fully applied")
    _close(noise.enabled[selected], [profiles[row]["noise_enabled"] for row in selected.tolist()],
           "sensor_noise_enabled", atol=0.)
    _close(noise.obs_lag[selected], torch.zeros_like(noise.obs_lag[selected]), "legacy_observation_lag", atol=0.)
    _close(noise.act_lag[selected], torch.zeros_like(noise.act_lag[selected]), "legacy_action_lag", atol=0.)
    _close(env.motor_strength[selected], ages.new_tensor([profiles[row]["motor_strength"] for row in selected.tolist()])[:, None].expand(-1, 6),
           "motor_strength")
    _close(env.spring_strength[selected], ages.new_tensor([profiles[row]["spring_strength"] for row in selected.tolist()])[:, None],
           "spring_strength")
    if getattr(getattr(env, "skills", None), "schedule", None) is not None:
        _close(env.skills.schedule.motor_scale[selected], torch.ones_like(env.motor_strength[selected]), "training_motor_scale")
    _verify_static(env, cases, selected)
    records = []
    for row in selected.tolist():
        records.append({"row": row, "case": cases[row]["name"], "base_case": profiles[row]["base_case"],
            "profile": profiles[row]["name"], "signal_age_ms": (delay.lags[row].float() * delay.dt_ms).cpu().tolist(),
            "noise_enabled": bool(noise.enabled[row]), "motor_strength": env.motor_strength[row].cpu().tolist(),
            "spring_strength": float(env.spring_strength[row, 0]),
            "base_mass_scale": float(env.dynamics_randomization.mass_scale[row, env.dynamics_randomization.base_id]),
            "com_offset_m": env.dynamics_randomization.com_offset[row].cpu().tolist(),
            "friction": float(env.contact_domain.mu[row]), "downlink_enabled": bool(env.command_transport.enabled[row]),
            "downlink_delay_ms": float(env.command_transport.base_delay[row] * 1000.) if env.command_transport.enabled[row] else 0.})
    return {"version": VERSION, "scope": declaration["scope"], "physics_dt": env.dt,
            "delay_curriculum_strength": float(delay.strength), "evaluation_actor_update": end,
            "sensor_noise_std_scaled35": expected_noise.cpu().tolist(), "verified_rows": records}

"""Left/right weight sharing must follow the physical mirror, not native signs.

Native right axes are mirrored (axis -y, nominal R = -L). Sharing decoders in
native coordinates makes identical native outputs, i.e. physically opposite
legs/wheels (scissor stance, counter-rotating wheels), the default function.
Synthetic tensors only; no simulator or checkpoint is used.
"""
import io

import pytest
import torch

from transformer_rl.formal.policy import (
    StructuredGaussianActor, StructuredPolicy, StructuredPolicyConfig, policy_config_from_dict,
)

PAIRS = ((10, 12), (11, 13), (16, 18), (17, 19), (22, 24), (23, 25), (20, 21), (26, 27))


def mirror_frames(x):
    """Physical reflection y -> -y of a [B, 5, 35] window."""
    y = x.clone()
    for index in (1, 2, 4, 6, 8):  # vy, yaw command, roll/yaw gyro, gravity y
        y[..., index] = -x[..., index]
    for a, b in PAIRS:
        y[..., a], y[..., b] = -x[..., b], -x[..., a]
    return y


def mirror_action(a):
    return torch.stack((-a[:, 2], -a[:, 3], -a[:, 0], -a[:, 1], -a[:, 5], -a[:, 4]), -1)


def native_swap_frames(x):
    y = x.clone()
    for index in (2, 4, 6, 8):
        y[..., index] = -x[..., index]
    for a, b in PAIRS:
        y[..., a], y[..., b] = x[..., b], x[..., a]
    return y


def windows(batch=512, seed=3):
    generator = torch.Generator().manual_seed(seed)
    x = torch.zeros(batch, 5, 35)
    x[..., 0] = torch.empty(batch, 1).uniform_(-1, 1, generator=generator)
    x[..., 2] = torch.empty(batch, 1).uniform_(-1, 1, generator=generator)
    x[..., 3] = 1.6
    x[..., 4:7] = .15 * torch.randn(batch, 5, 3, generator=generator)
    x[..., 7:9] = .05 * torch.randn(batch, 5, 2, generator=generator)
    x[..., 9] = -1
    x[..., 10:14] = .2 * torch.randn(batch, 5, 4, generator=generator)
    x[..., 16:22] = .1 * torch.randn(batch, 5, 6, generator=generator)
    x[..., 22:28] = torch.randn(batch, 5, 6, generator=generator)
    x[..., 28] = 1
    x[..., 34] = 5
    return x


def scaled(policy):
    # Initial heads are scaled by 0.01; enlarge them so the test measures the
    # architecture's symmetry rather than near-zero outputs.
    with torch.no_grad():
        for head in (policy.leg_decoder[-1], policy.wheel_decoder[-1]):
            head.weight.mul_(100)
            head.bias.mul_(100)
    return policy.eval()


def equivariance(policy, mirror, action_map):
    x = windows()
    with torch.no_grad():
        a = policy(x)
        return ((policy(mirror(x)) - action_map(a)).norm(dim=-1).mean() / a.norm(dim=-1).mean()).item()


def test_legacy_policy_keeps_native_sharing_and_its_exact_state_dict_keys():
    torch.manual_seed(0)
    legacy = StructuredPolicy(StructuredPolicyConfig())
    torch.manual_seed(0)
    mirrored = StructuredPolicy(StructuredPolicyConfig(mirror_canonical=True))
    assert legacy.state_dict().keys() == mirrored.state_dict().keys()
    assert not any('mirror' in key or 'side_sign' in key for key in legacy.state_dict())
    legacy, mirrored = scaled(legacy), scaled(mirrored)
    # Native sharing is (almost) equivariant to the sign-free swap: the bias
    # that produced the scissor stance. The mirrored frame removes it.
    assert equivariance(legacy, native_swap_frames, lambda a: a[:, [2, 3, 0, 1, 5, 4]]) < .3
    assert equivariance(legacy, mirror_frames, mirror_action) > 1.5
    assert equivariance(mirrored, mirror_frames, mirror_action) < .3
    assert equivariance(mirrored, native_swap_frames, lambda a: a[:, [2, 3, 0, 1, 5, 4]]) > 1.5


def test_symmetric_state_gives_nearly_physically_symmetric_action_with_equal_side_identities():
    torch.manual_seed(1)
    policy = scaled(StructuredPolicy(StructuredPolicyConfig(mirror_canonical=True)))
    with torch.no_grad():
        policy.body_identity[3].copy_(policy.body_identity[2])
        policy.body_identity[5].copy_(policy.body_identity[4])
        policy.readout.query_identity[1].copy_(policy.readout.query_identity[0])
        policy.readout.query_identity[3].copy_(policy.readout.query_identity[2])
    x = windows(64)
    x[..., 2] = 0  # no yaw command
    x[..., 4] = x[..., 6] = x[..., 8] = 0  # no roll/yaw rate, no lateral gravity
    for a, b in PAIRS:
        x[..., b] = -x[..., a]
    with torch.no_grad():
        action = policy(x)
    # Only the untrained state estimate (vy, contact pair) is not mirror
    # symmetric, so the physical-mirror residual is small but not zero.
    residual = (action - mirror_action(action)).abs().mean() / action.abs().mean()
    assert residual < .05
    # Symmetric motion: native wheel targets of the two sides have opposite signs.
    assert (torch.sign(action[:, 4]) == -torch.sign(action[:, 5])).float().mean() > .9


def test_mirror_policy_scripts_and_checkpoint_config_tolerates_legacy_omission():
    policy = StructuredPolicy(StructuredPolicyConfig(mirror_canonical=True)).for_deployment()
    scripted = torch.jit.script(policy)
    x = windows(4)
    torch.testing.assert_close(scripted(x), policy(x))
    stream = io.BytesIO()
    torch.jit.save(scripted, stream)
    legacy = {key: value for key, value in vars(StructuredPolicyConfig()).items()
              if key not in ('mirror_canonical', 'initial_leg_std', 'initial_wheel_std')}
    restored = policy_config_from_dict(legacy, complete=True)
    assert restored.mirror_canonical is False and restored.initial_leg_std == .3
    with pytest.raises(ValueError):
        policy_config_from_dict({key: value for key, value in legacy.items() if key != 'body_width'}, complete=True)


def test_initial_exploration_std_is_configured_per_actuator_group():
    actor = StructuredGaussianActor(StructuredPolicyConfig(initial_leg_std=.5, initial_wheel_std=.4))
    torch.testing.assert_close(actor.log_std.exp(), torch.tensor([.5] * 4 + [.4] * 2))
    with pytest.raises(ValueError):
        StructuredPolicyConfig(initial_leg_std=0.)

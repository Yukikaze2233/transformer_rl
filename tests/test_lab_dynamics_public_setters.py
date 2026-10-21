"""GA property-cache coherence without changing the startup parameter draw."""
from types import SimpleNamespace
import math

import pytest
import torch

from test_lab_dynamics import native
from transformer_rl.formal.lab_env.dynamics import LabStartupDynamics
from transformer_rl.formal.simulation.dynamics import StartupDynamicsConfig


class _StaticComCache:
    """Model GA's static COM getter: time advancement alone does not refresh it."""

    def __init__(self, view):
        self.view = view
        self._value = None
        self.invalidations = 0

    @property
    def body_com_pose_b(self):
        if self._value is None:
            self._value = self.view.local_com.clone()
        return SimpleNamespace(torch=self._value)

    def publish(self, value):
        self.body_com_pose_b.torch.copy_(value)
        self.invalidations += 1


def public_native():
    robot, binding, view, writes = native()
    # A nonidentity COM orientation catches an accidental xyzw/wxyz shuffle.
    view.local_com[..., 3:] = torch.tensor([.1, .2, .3, math.sqrt(.86)])
    robot.backend_body_names = tuple(robot.body_names)
    robot.data = _StaticComCache(view)
    calls = []
    methods = (('set_masses_index', 'masses', 'set_masses'),
               ('set_coms_index', 'coms', 'set_coms'),
               ('set_inertias_index', 'inertias', 'set_inertias'))
    for method, argument, raw_method in methods:
        def write(*, env_ids=None, body_ids=None, full_data=False,
                  argument=argument, raw_method=raw_method, method=method, **kwargs):
            assert set(kwargs) == {argument}
            value = kwargs[argument]
            assert value.dtype == torch.float32 and value.is_contiguous()
            assert value.device.type == 'cpu'
            calls.append((method, env_ids, body_ids, full_data, value.clone()))
            assert env_ids is None and body_ids is None and full_data is True
            getattr(view, raw_method)(value, torch.arange(view.count, dtype=torch.int32))
            if argument == 'coms':
                robot.data.publish(value)
        setattr(robot, method, write)
    return robot, binding, view, writes, calls


def apply(robot, binding, view, config, *, use_public_setters, tensor_to_native=None):
    return LabStartupDynamics(
        robot, binding, view.count, float(view.masses[0].sum()), 31, config,
        device='cpu', native_to_tensor=lambda value: value,
        tensor_to_native=(lambda value: value) if tensor_to_native is None else tensor_to_native,
        use_public_setters=use_public_setters)


def test_public_setters_refresh_a_preheated_com_cache_and_preserve_the_raw_draw():
    legacy_robot, legacy_binding, legacy_view, _, legacy_calls = public_native()
    ga_robot, ga_binding, ga_view, _, ga_calls = public_native()
    old_cache = legacy_robot.data.body_com_pose_b.torch.clone()
    torch.testing.assert_close(ga_robot.data.body_com_pose_b.torch, old_cache, rtol=0, atol=0)
    before = torch.get_rng_state().clone()
    configuration = StartupDynamicsConfig(nominal_fraction=0)

    legacy = apply(legacy_robot, legacy_binding, legacy_view, configuration, use_public_setters=False)
    current = apply(ga_robot, ga_binding, ga_view, configuration, use_public_setters=True)

    assert torch.equal(before, torch.get_rng_state())
    assert not legacy_calls
    assert [call[0] for call in ga_calls] == [
        'set_masses_index', 'set_coms_index', 'set_inertias_index', 'set_coms_index']
    assert ga_robot.data.invalidations == 2
    # This is the discriminating assertion: native readback alone passes on
    # the legacy path, even though the preheated public cache remains stale.
    assert torch.equal(legacy_robot.data.body_com_pose_b.torch, old_cache)
    assert not torch.equal(old_cache[:, 0, :3], legacy_view.local_com[:, 0, :3])
    assert torch.equal(ga_robot.data.body_com_pose_b.torch, ga_view.local_com)
    for field in ('masses', 'local_com', 'inertias'):
        assert torch.equal(getattr(legacy_view, field), getattr(ga_view, field))
    assert torch.equal(ga_view.local_com[..., 3:], old_cache[..., 3:])
    assert legacy.receipt['native_parameters_sha256'] == current.receipt['native_parameters_sha256']
    assert current.receipt['property_write_api'] == 'articulation_public_index_setters'
    assert 'property_write_api' not in legacy.receipt


def test_public_all_row_writes_do_not_use_the_legacy_native_converter():
    robot, binding, view, writes, calls = public_native()

    def reject_legacy_conversion(value):
        raise AssertionError('public setter inputs must stay explicit torch tensors')

    apply(robot, binding, view, StartupDynamicsConfig(), use_public_setters=True,
          tensor_to_native=reject_legacy_conversion)
    assert writes == ['masses', 'local_com', 'inertias', 'local_com']
    assert [call[4].shape for call in calls] == [(2, 19), (2, 19, 7), (2, 19, 9), (2, 19, 7)]
    assert all(call[1:4] == (None, None, True) for call in calls)


def test_public_setter_contract_is_checked_before_any_partial_property_write():
    robot, binding, view, writes, calls = public_native()
    del robot.set_inertias_index
    with pytest.raises(ValueError, match='all public articulation setters'):
        apply(robot, binding, view, StartupDynamicsConfig(), use_public_setters=True)
    assert not writes and not calls


def test_public_setters_reject_a_native_public_body_order_mismatch_before_writing():
    robot, binding, view, writes, calls = public_native()
    robot.backend_body_names = tuple(reversed(robot.body_names))
    with pytest.raises(ValueError, match='public body ordering'):
        apply(robot, binding, view, StartupDynamicsConfig(), use_public_setters=True)
    assert not writes and not calls


def test_public_com_readback_rejects_a_unit_length_quaternion_permutation():
    robot, binding, view, _, _ = public_native()
    original = robot.set_coms_index

    def corrupt_com_write(**kwargs):
        original(**kwargs)
        view.local_com[..., 3:] = view.local_com[..., 3:].roll(1, dims=-1)

    robot.set_coms_index = corrupt_com_write
    with pytest.raises(RuntimeError, match='position or xyzw rotation'):
        apply(robot, binding, view, StartupDynamicsConfig(), use_public_setters=True)


def test_public_com_write_accepts_independent_quaternion_sign_changes():
    robot, binding, view, _, _ = public_native()
    original = robot.set_coms_index

    def change_signs(**kwargs):
        values = kwargs['coms'].clone()
        values[0, ::2, 3:] *= -1
        original(**{**kwargs, 'coms': values})

    robot.set_coms_index = change_signs
    apply(robot, binding, view, StartupDynamicsConfig(), use_public_setters=True)
    assert torch.equal(robot.data.body_com_pose_b.torch, view.local_com)


def reselected_native():
    robot, binding, view, writes, calls = public_native()
    view.local_com[..., 3:] = torch.tensor([0., 0., 0., 1.])
    view.inertias.copy_(torch.diag(torch.tensor([.01, .02, .03])).reshape(1, 1, 9).expand(2, 19, 9))
    original = robot.set_inertias_index
    boundaries = []

    def reselect_principal_axes(**kwargs):
        original(**kwargs)
        # A 180-degree principal-axis sign change preserves this anisotropic
        # link-frame inertia. GA's inertia setter leaves the static cache old.
        view.local_com[..., 3:] = torch.tensor([0., 0., 1., 0.])
        boundaries.append((view.local_com.clone(), robot.data.body_com_pose_b.torch.clone(),
                           view.inertias.clone()))

    robot.set_inertias_index = reselect_principal_axes
    return robot, binding, view, writes, calls, boundaries


def test_principal_frame_reselection_preserves_physics_and_refreshes_the_static_cache():
    robot, binding, view, writes, calls, boundaries = reselected_native()
    robot.data.body_com_pose_b  # Deliberately prewarm the static GA cache.
    original_mass, original_inertia = view.masses.clone(), view.inertias.clone()
    before = torch.get_rng_state().clone()

    result = apply(robot, binding, view, StartupDynamicsConfig(nominal_fraction=0), use_public_setters=True)

    assert torch.equal(before, torch.get_rng_state())
    native_after_inertia, cache_after_inertia, inertia_after_write = boundaries[0]
    assert not torch.equal(native_after_inertia[..., 3:], cache_after_inertia[..., 3:])
    assert torch.equal(calls[-1][4], native_after_inertia)
    assert torch.equal(view.local_com, native_after_inertia)
    assert torch.equal(robot.data.body_com_pose_b.torch, native_after_inertia)
    assert torch.equal(view.inertias, inertia_after_write)
    torch.testing.assert_close(view.inertias, original_inertia * (view.masses / original_mass)[..., None])
    assert writes == ['masses', 'local_com', 'inertias', 'local_com']
    assert result.receipt['status'] == 'native_readback_matched'


@pytest.mark.parametrize('field', ['inertias', 'local_com'])
def test_principal_frame_reselection_does_not_hide_wrong_inertia_or_com_translation(field):
    robot, binding, view, _, calls, _ = reselected_native()
    original = robot.set_inertias_index

    def corrupt(**kwargs):
        original(**kwargs)
        getattr(view, field)[0, 0, 0] += .1

    robot.set_inertias_index = corrupt
    with pytest.raises(RuntimeError, match='post-write readback'):
        apply(robot, binding, view, StartupDynamicsConfig(), use_public_setters=True)
    assert len(calls) == 3  # Invalid physics must not be republished as accepted cache data.


@pytest.mark.parametrize('field', ['masses', 'inertias', 'local_com'])
def test_final_cache_publication_must_not_change_native_mechanical_parameters(field):
    robot, binding, view, _, _, _ = reselected_native()
    original = robot.set_coms_index
    count = 0

    def corrupt_final_publication(**kwargs):
        nonlocal count
        original(**kwargs)
        count += 1
        if count == 2:
            value = getattr(view, field)
            if field == 'masses':
                value[0, 0] += .1
            else:
                value[0, 0, 0] += .1

    robot.set_coms_index = corrupt_final_publication
    with pytest.raises(RuntimeError, match='cache publication changed native'):
        apply(robot, binding, view, StartupDynamicsConfig(), use_public_setters=True)


def test_final_cache_publication_rejects_a_stale_public_pose():
    robot, binding, view, _, _, _ = reselected_native()
    original = robot.data.publish
    count = 0

    def leave_final_cache_stale(value):
        nonlocal count
        count += 1
        if count == 1:
            original(value)

    robot.data.publish = leave_final_cache_stale
    with pytest.raises(RuntimeError, match='cache differs from the final native principal frame'):
        apply(robot, binding, view, StartupDynamicsConfig(), use_public_setters=True)


def test_nominal_public_path_does_not_write_or_consume_global_rng():
    robot, binding, view, writes, calls = public_native()
    expected = robot.data.body_com_pose_b.torch.clone()
    before = torch.get_rng_state().clone()
    result = apply(robot, binding, view, None, use_public_setters=True)
    assert torch.equal(before, torch.get_rng_state())
    assert not writes and not calls
    assert torch.equal(robot.data.body_com_pose_b.torch, expected)
    assert not result.receipt['randomization_active']


def test_property_write_api_selection_requires_an_explicit_bool():
    robot, binding, view, writes, _ = public_native()
    with pytest.raises(ValueError, match='fresh native mechanical rows'):
        apply(robot, binding, view, StartupDynamicsConfig(), use_public_setters=1)
    assert not writes

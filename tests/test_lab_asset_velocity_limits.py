"""Pinned mechanical bounds must survive asset/export ordering and reject guesses."""
from copy import deepcopy
import hashlib
import json

import pytest

from transformer_rl.formal.lab_env.settings import AssetDescription


def asset(tmp_path, joints):
    raw = json.dumps({'joints': joints}).encode()
    (tmp_path / 'model_spec.json').write_bytes(raw)
    manifest = {'tree_joint_names': ['hip', 'slide'],
                'files_sha256': {'model_spec.json': hashlib.sha256(raw).hexdigest()}}
    return AssetDescription(tmp_path, manifest, tmp_path / 'robot.usda', tmp_path / 'robot.urdf',
                            None, .32, 'unused')


def source_joints():
    # The prismatic value stays in m/s; the rotary value stays in rad/s.
    return [{'name': 'slide', 'type': 'prismatic', 'limit': {'velocity': .7}},
            {'name': 'hip', 'type': 'continuous', 'limit': {'velocity': 32.}}]


def test_asset_limits_use_named_tree_order_without_unit_conversion(tmp_path):
    selected = asset(tmp_path, source_joints())
    assert selected.joint_velocity_limits() == {'hip': 32., 'slide': .7}
    assert list(selected.joint_velocity_limits()) == ['hip', 'slide']


def test_velocity_limit_source_must_match_the_frozen_asset_pin(tmp_path):
    selected = asset(tmp_path, source_joints())
    (tmp_path / 'model_spec.json').write_text('{}')
    with pytest.raises(ValueError, match='pin'):
        selected.joint_velocity_limits()


@pytest.mark.parametrize('velocity', [None, 0., -1., True, float('nan'), float('inf')])
def test_missing_or_invalid_velocity_is_not_an_unbounded_fallback(tmp_path, velocity):
    joints = source_joints()
    joints[1]['limit']['velocity'] = velocity
    with pytest.raises(ValueError, match='positive'):
        asset(tmp_path, joints).joint_velocity_limits()


@pytest.mark.parametrize('velocity', ['32', '3.2e1'])
def test_urdf_numeric_attribute_strings_keep_their_native_units(tmp_path, velocity):
    joints = source_joints()
    joints[1]['limit']['velocity'] = velocity
    assert asset(tmp_path, joints).joint_velocity_limits()['hip'] == 32.


@pytest.mark.parametrize('velocity', ['', 'unknown', 'NaN', 'Infinity'])
def test_nonfinite_or_missing_numeric_attribute_strings_are_rejected(tmp_path, velocity):
    joints = source_joints()
    joints[1]['limit']['velocity'] = velocity
    with pytest.raises(ValueError):
        asset(tmp_path, joints).joint_velocity_limits()


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'unknown', 'invalid_type'])
def test_asset_limits_cannot_silently_drop_or_relabel_a_joint(tmp_path, mutation):
    joints = deepcopy(source_joints())
    if mutation == 'duplicate':
        joints[1]['name'] = 'slide'
    elif mutation == 'missing':
        joints.pop()
    elif mutation == 'unknown':
        joints[1]['name'] = 'different_hip'
    else:
        joints[1]['type'] = 'fixed'
    with pytest.raises(ValueError):
        asset(tmp_path, joints).joint_velocity_limits()


def test_asset_limits_do_not_follow_an_alternate_model_symlink(tmp_path):
    selected = asset(tmp_path, source_joints())
    original = tmp_path / 'model_spec.json'
    original.rename(tmp_path / 'other.json')
    original.symlink_to(tmp_path / 'other.json')
    with pytest.raises(ValueError, match='regular'):
        selected.joint_velocity_limits()

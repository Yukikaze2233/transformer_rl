import pytest

from transformer_rl.formal.lab_env.contact_mapping import contact_body_mapping, required_contact_bodies


def test_required_bodies_include_base_and_both_wheels():
    names = ["base_link", "L_link3", "R_link3", "LL_link4", "RL_link4"]
    assert required_contact_bodies(names, [1, 2]) == ("L_link3", "R_link3", "base_link")


def test_mapping_rejects_missing_required_body():
    with pytest.raises(RuntimeError, match="L_link3"):
        contact_body_mapping(["base_link", "L_link3", "R_link3"], ["base_link", "R_link3"], [1, 2])


def test_mapping_is_one_to_one_and_preserves_native_order():
    source, target, required = contact_body_mapping(
        ["base_link", "L_link3", "R_link3", "LL_link4"],
        ["R_link3", "base_link", "L_link3"], [1, 2])
    assert required == ("L_link3", "R_link3", "base_link")
    assert target == [0, 1, 2]
    assert source == [1, 2, 0]


def test_formal_reward_requires_leg_contacts_instead_of_silently_zero_filling():
    names = ['base_link', 'L_link3', 'R_link3', 'LL_link4']
    with pytest.raises(RuntimeError, match='LL_link4'):
        contact_body_mapping(names, names[:3], [1, 2], require_all=True)
    source, target, required = contact_body_mapping(names, names[::-1], [1, 2], require_all=True)
    assert required == tuple(names)
    assert target == list(range(4)) and source == [3, 2, 1, 0]


@pytest.mark.parametrize('wheels', ([1, 1], [-1, 2], [1, 9]))
def test_invalid_wheel_topology_is_rejected(wheels):
    with pytest.raises(ValueError, match='topology'):
        required_contact_bodies(['base_link', 'L_link3', 'R_link3'], wheels)


def test_duplicate_sensor_names_are_not_arbitrarily_mapped():
    with pytest.raises(RuntimeError, match='ambiguous'):
        contact_body_mapping(['base_link', 'L_link3', 'R_link3'],
                             ['base_link', 'L_link3', 'R_link3', 'L_link3'], [1, 2])

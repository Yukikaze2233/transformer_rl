import math

import pytest
import torch

from transformer_rl.reward_components import RewardComponentStatistics


def test_vector_steps_preserve_density_and_actual_step_units():
    stats = RewardComponentStatistics(2, .25)
    stats.observe(torch.tensor([1., -2.]),
                  {"height": torch.tensor([4., -4.]), "smooth": torch.tensor([-2., -6.])})
    stats.observe(torch.tensor([3., 0.]),
                  {"smooth": torch.tensor([0., -8.]), "height": torch.tensor([8., 0.])})
    result = stats.drain()
    assert (result["steps"], result["samples"]) == (2, 4)
    total = result["actual_total_reward"]
    assert total["unit"] == "reward/step"
    assert total["sum"] == 2
    assert total["mean"] == .5
    assert total["rms"] == pytest.approx(math.sqrt(3.5))
    assert (total["min"], total["max"]) == (-2, 3)
    height = result["continuous_components"]["terms"]["height"]
    assert height["density"]["unit"] == "reward/s"
    assert height["density"]["sum"] == 8
    assert height["density"]["mean"] == 2
    assert height["density"]["rms"] == pytest.approx(math.sqrt(24))
    assert (height["density"]["min"], height["density"]["max"]) == (-4, 8)
    assert height["step_reward"]["sum"] == 2
    assert height["step_reward"]["mean"] == .5
    assert height["step_reward"]["rms"] == pytest.approx(math.sqrt(1.5))
    assert (height["step_reward"]["min"], height["step_reward"]["max"]) == (-1, 2)
    residual = result["residual_step_reward"]["statistics"]
    # The four residual contributions are .5, .5, 1 and 2.
    assert residual["sum"] == 4
    assert residual["mean"] == 1
    assert residual["rms"] == pytest.approx(math.sqrt(1.375))
    assert (residual["min"], residual["max"]) == (.5, 2)
    assert height["density"]["samples"] == residual["samples"] == 4
    assert height["density"]["steps"] == residual["steps"] == 2


def test_only_explicit_events_receive_event_attribution():
    stats = RewardComponentStatistics(2, .1)
    stats.observe(torch.tensor([.2, -4.8], dtype=torch.float64),
                  {"alive": torch.tensor([2., 2.], dtype=torch.float64)},
                  {"terminal": torch.tensor([0., -5.], dtype=torch.float64)})
    result = stats.drain()
    event = result["event_components"]["terms"]["terminal"]
    assert result["event_components"]["available"] is True
    assert event["unit"] == "reward/step"
    assert event["sum"] == -5
    assert event["mean"] == -2.5
    assert event["rms"] == pytest.approx(math.sqrt(12.5))
    assert result["residual_step_reward"]["statistics"]["rms"] == pytest.approx(0, abs=1e-14)
    assert result["residual_step_reward"]["interpretation"].startswith("unattributed_terms_after")

    stats.observe(torch.tensor([.2, -4.8]), {"alive": torch.tensor([2., 2.])})
    without_events = stats.drain()
    assert without_events["event_components"]["available"] is False
    assert without_events["event_components"]["terms"] is None
    assert without_events["residual_step_reward"]["interpretation"] == (
        "unattributed_events_and_omitted_continuous_terms")
    assert without_events["residual_step_reward"]["statistics"]["sum"] == pytest.approx(-5)


def test_residual_retains_omitted_continuous_reward_even_with_explicit_events():
    stats = RewardComponentStatistics(1, .1)
    stats.observe(torch.tensor([.7], dtype=torch.float64),
                  {"reported": torch.tensor([2.], dtype=torch.float64)},
                  {"explicit_event": torch.tensor([.3], dtype=torch.float64)})
    residual = stats.drain()["residual_step_reward"]
    assert residual["available"] is True
    assert residual["statistics"]["sum"] == pytest.approx(.2)
    assert "unattributed" in residual["interpretation"]
    assert "error" not in residual["interpretation"]


def test_unavailable_continuous_source_does_not_manufacture_zero_terms():
    stats = RewardComponentStatistics(2, .1)
    stats.observe(torch.tensor([1., 3.]), None)
    stats.observe(torch.tensor([-2., 2.]), None)
    result = stats.drain()
    assert result["actual_total_reward"]["sum"] == 4
    assert result["actual_total_reward"]["mean"] == 1
    assert result["continuous_components"]["available"] is False
    assert result["continuous_components"]["terms"] is None
    assert result["residual_step_reward"]["available"] is False
    assert result["residual_step_reward"]["statistics"] is None
    assert result["event_components"]["available"] is False


def test_event_source_can_be_available_when_continuous_source_is_unavailable():
    stats = RewardComponentStatistics(1, .01)
    stats.observe(torch.tensor([4.]), None, {"event": torch.tensor([3.])})
    result = stats.drain()
    assert result["event_components"]["available"] is True
    assert result["event_components"]["terms"]["event"]["mean"] == 3
    assert result["residual_step_reward"]["available"] is False
    assert result["residual_step_reward"]["statistics"] is None


def test_explicit_empty_sources_are_available_and_do_not_imply_gate_coverage():
    stats = RewardComponentStatistics(1, .01)
    stats.observe(torch.tensor([2.]), {}, {})
    result = stats.drain()
    assert result["continuous_components"]["available"] is True
    assert result["continuous_components"]["terms"] == {}
    assert result["event_components"]["available"] is True
    assert result["event_components"]["terms"] == {}
    assert result["residual_step_reward"]["statistics"]["mean"] == 2
    assert result["gate_coverage"]["available"] is False
    assert "explicit_gate" in result["gate_coverage"]["reason"]
    stats.observe(torch.zeros(1), {"zero": torch.zeros(1)})
    zeros = stats.drain()
    assert zeros["continuous_components"]["terms"]["zero"]["density"]["mean"] == 0
    assert zeros["gate_coverage"]["available"] is False


def test_empty_and_repeated_drains_are_truthfully_empty_and_allow_new_schema():
    stats = RewardComponentStatistics(2, .25)
    empty = stats.drain()
    assert empty == stats.drain()
    assert empty["actual_total_reward"]["available"] is False
    assert empty["actual_total_reward"]["sum"] is None
    assert empty["continuous_components"]["terms"] is None
    assert (empty["steps"], empty["samples"]) == (0, 0)
    stats.observe(torch.tensor([1., 3.]), {"first": torch.tensor([2., 2.])})
    first = stats.drain()
    first["continuous_components"]["terms"]["first"]["density"]["sum"] = 999
    stats.observe(torch.tensor([5., 7.]), None, {"second": torch.tensor([0., 2.])})
    second = stats.drain()
    assert (second["steps"], second["samples"]) == (1, 2)
    assert second["actual_total_reward"]["mean"] == 6
    assert second["continuous_components"]["available"] is False
    assert second["event_components"]["terms"]["second"]["mean"] == 1
    assert stats.drain() == empty


@pytest.mark.parametrize("num_envs", [0, -1, 1., True, None, "1"])
def test_invalid_population_rejected(num_envs):
    with pytest.raises(ValueError, match="positive integer"):
        RewardComponentStatistics(num_envs, .01)


@pytest.mark.parametrize("dt", [0, -1, True, None, "0.01", float("inf"), float("nan"), 10**400])
def test_invalid_timestep_rejected(dt):
    with pytest.raises(ValueError, match="finite and positive"):
        RewardComponentStatistics(1, dt)


@pytest.mark.parametrize("field", ["total", "continuous", "event"])
@pytest.mark.parametrize("invalid", ["shape", "integer", "boolean", "complex", "meta_device", "nan", "inf"])
def test_invalid_values_rejected_without_discarding_valid_window(field, invalid):
    stats = RewardComponentStatistics(2, .25)
    stats.observe(torch.ones(2), {"term": torch.ones(2)}, {"event": torch.zeros(2)})
    bad = {
        "shape": lambda: torch.ones(2, 1),
        "integer": lambda: torch.ones(2, dtype=torch.int64),
        "boolean": lambda: torch.ones(2, dtype=torch.bool),
        "complex": lambda: torch.ones(2, dtype=torch.complex64),
        "meta_device": lambda: torch.empty(2, device="meta"),
        "nan": lambda: torch.tensor([0., float("nan")]),
        "inf": lambda: torch.tensor([0., float("inf")]),
    }[invalid]()
    total, continuous, events = torch.ones(2), {"term": torch.ones(2)}, {"event": torch.zeros(2)}
    if field == "total":
        total = bad
    elif field == "continuous":
        continuous["term"] = bad
    else:
        events["event"] = bad
    with pytest.raises((ValueError, FloatingPointError)):
        stats.observe(total, continuous, events)
    stats.observe(torch.full((2,), 3.), {"term": torch.ones(2)}, {"event": torch.zeros(2)})
    result = stats.drain()
    assert result["steps"] == 2
    assert result["actual_total_reward"]["sum"] == 8


@pytest.mark.parametrize("field", ["continuous", "event"])
@pytest.mark.parametrize("invalid", [[], 2, {"": torch.ones(1)}, {" padded": torch.ones(1)},
                                    {"line\nbreak": torch.ones(1)}, {3: torch.ones(1)}, {"term": 1.}])
def test_invalid_component_metadata_rejected(field, invalid):
    stats = RewardComponentStatistics(1, .1)
    with pytest.raises(ValueError):
        if field == "continuous":
            stats.observe(torch.ones(1), invalid)
        else:
            stats.observe(torch.ones(1), {}, invalid)
    assert stats.drain()["steps"] == 0


@pytest.mark.parametrize("first,second", [(None, {}), ({}, None), ({}, {"new": torch.ones(1)}),
                                         ({"old": torch.ones(1)}, {}),
                                         ({"old": torch.ones(1)}, {"new": torch.ones(1)})])
@pytest.mark.parametrize("field", ["continuous", "event"])
def test_source_availability_and_term_schema_cannot_change_midwindow(first, second, field):
    stats = RewardComponentStatistics(1, .1)
    if field == "continuous":
        stats.observe(torch.ones(1), first)
        with pytest.raises(ValueError, match="availability or keys changed"):
            stats.observe(torch.ones(1), second)
    else:
        stats.observe(torch.ones(1), {}, first)
        with pytest.raises(ValueError, match="availability or keys changed"):
            stats.observe(torch.ones(1), {}, second)
    assert stats.drain()["steps"] == 1


def test_pre_reset_input_aliases_and_gradients_are_not_retained_or_mutated():
    stats = RewardComponentStatistics(2, .25)
    total = torch.tensor([1., 3.], dtype=torch.float64, requires_grad=True)
    density = torch.tensor([2., 6.], dtype=torch.float64, requires_grad=True)
    event = torch.tensor([0., 1.], dtype=torch.float64, requires_grad=True)
    originals = [value.detach().clone() for value in (total, density, event)]
    stats.observe(total, {"term": density}, {"event": event})
    for value, original in zip((total, density, event), originals):
        torch.testing.assert_close(value.detach(), original, rtol=0, atol=0)
        assert value.grad is None
    assert stats._moments.grad_fn is None
    assert stats._moments.requires_grad is False
    with torch.no_grad():
        total.zero_()
        density.zero_()
        event.zero_()
    result = stats.drain()
    assert result["actual_total_reward"]["sum"] == 4
    assert result["continuous_components"]["terms"]["term"]["density"]["sum"] == 8
    assert result["event_components"]["terms"]["event"]["sum"] == 1
    assert result["residual_step_reward"]["statistics"]["sum"] == 1
    (total.sum() + density.sum() + event.sum()).backward()
    for value in (total, density, event):
        torch.testing.assert_close(value.grad, torch.ones(2, dtype=torch.float64))


def test_device_alias_and_noncontiguous_sources_use_actual_tensor_device():
    stats = RewardComponentStatistics(2, .1)
    source = torch.tensor([1., 100., 3., 100.], device="cpu:0")
    stats.observe(source[::2], {"term": torch.tensor([2., 4.], device="cpu")})
    stats.observe(torch.tensor([5., 7.], device="cpu"), {"term": torch.tensor([6., 8.], device="cpu:0")})
    result = stats.drain()
    assert result["actual_total_reward"]["sum"] == 16
    assert result["continuous_components"]["terms"]["term"]["density"]["mean"] == 5


def test_streaming_state_size_does_not_grow_with_steps():
    stats = RewardComponentStatistics(3, .01)
    for _ in range(100):
        stats.observe(torch.ones(3), {"a": torch.ones(3), "b": torch.zeros(3)},
                      {"event": torch.zeros(3)})
        assert stats._moments.shape == (7, 4)
    result = stats.drain()
    assert (result["steps"], result["samples"]) == (100, 300)
    assert result["actual_total_reward"]["sum"] == 300
    assert "all_returned_rows" in result["scope"]
    assert not torch.cuda.is_initialized()


def test_finite_source_with_overflowing_moments_rejected_atomically():
    stats = RewardComponentStatistics(1, .1)
    stats.observe(torch.ones(1), {})
    with pytest.raises(FloatingPointError, match="moments are nonfinite"):
        stats.observe(torch.tensor([1e200], dtype=torch.float64), {})
    assert stats.drain()["actual_total_reward"]["sum"] == 1


def test_scaled_density_overflow_is_rejected_before_reporting():
    stats = RewardComponentStatistics(1, 1e300)
    with pytest.raises(FloatingPointError):
        stats.observe(torch.zeros(1, dtype=torch.float64), {"term": torch.tensor([1e10], dtype=torch.float64)})
    assert stats.drain()["steps"] == 0


@pytest.mark.parametrize("total", [[1., 2.], None,
                                  torch.sparse_coo_tensor(torch.tensor([[0, 1]]),
                                                          torch.tensor([1., 2.]), (2,),
                                                          check_invariants=True)])
def test_non_tensor_and_sparse_rewards_rejected(total):
    stats = RewardComponentStatistics(2, .01)
    with pytest.raises(ValueError):
        stats.observe(total, None)
    assert stats.drain()["actual_total_reward"]["available"] is False

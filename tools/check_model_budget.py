"""Audit executable actors against the declared deployment parameter budgets.

Counts exclude Gaussian exploration parameters and training-only networks.
This CPU construction audit makes no latency or control-quality claims.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch
from torch import nn

from transformer_rl.config import load_config
from transformer_rl.model import ActorCritic


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def audit(protocol_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    common = protocol["common"]
    capacity = protocol["model_capacity"]
    base_path = (protocol_path.parent / common["base_config"]).resolve()
    base, _, _ = load_config(base_path)
    reference_path = (protocol_path.parent / protocol["reference_cli_spec"]).resolve()
    reference = json.loads(reference_path.read_text())
    references = {item["name"]: item for item in reference["variants"]}
    full_path = (protocol_path.parent / protocol["full_cli_spec"]).resolve()
    compiled = {item["name"]: item for item in json.loads(full_path.read_text())["variants"]}
    assert base.frame_dim == common["frame_dim"]
    assert base.history_length == common["history_length"]
    assert base.action_dim == common["action_dim"]
    assert base.critic_hidden == tuple(common["critic_hidden"])
    assert base.critic_dim == common["critic_dim"]
    for key, value in capacity["transformer_profiles"]["standard"].items():
        assert common["estimator_transformer"][key] == value
    assert common["estimator_mlp_hidden"] == capacity["mlp_profiles"]["standard"]["estimator_hidden"]
    results = []
    for variant in protocol["variants"]:
        for profile in ("standard", "compact"):
            transformer = capacity["transformer_profiles"][profile]
            mlp_profile = capacity["mlp_profiles"][profile]
            model_config = replace(
                base, d_model=transformer["d_model"],
                num_layers=transformer["layers"], num_heads=transformer["heads"],
                ffn_dim=transformer["ffn_dim"], residual_type="add",
                time_encoding="index", readout_type="last",
            )
            training_only = 0
            if variant["family"] == "direct":
                overrides = dict(references[variant["name"]]["model"])
                if "baseline_hidden" in overrides:
                    overrides["baseline_hidden"] = tuple(overrides["baseline_hidden"])
                direct_config = replace(base, **overrides)
                if variant["backbone"] == "mlp":
                    if profile == "standard":
                        assert direct_config.baseline_hidden == tuple(mlp_profile["direct_hidden"])
                    direct_config = replace(direct_config, baseline_hidden=tuple(mlp_profile["direct_hidden"]))
                else:
                    if profile == "standard":
                        for field in ("d_model", "num_layers", "num_heads", "ffn_dim"):
                            assert getattr(direct_config, field) == getattr(model_config, field)
                    direct_config = replace(
                        direct_config, d_model=model_config.d_model,
                        num_layers=model_config.num_layers, num_heads=model_config.num_heads,
                        ffn_dim=model_config.ffn_dim,
                    )
                actor = ActorCritic(direct_config).actor
                total = parameter_count(actor) - actor.log_std.numel()
                if variant["backbone"] == "mlp":
                    controller = parameter_count(actor.network)
                else:
                    controller = parameter_count(actor.mean_head)
                encoder = total - controller
                basis = "instantiated_existing_actor"
            else:
                overrides = dict(compiled[variant["name"]]["model"])
                for key in ("state_indices", "baseline_hidden", "controller_hidden"):
                    if key in overrides:
                        overrides[key] = tuple(overrides[key])
                estimator_config = replace(base, **overrides)
                assert estimator_config.estimator_type == variant["family"]
                assert len(estimator_config.state_indices) == len(variant["state_targets"])
                if variant["backbone"] == "mlp":
                    estimator_config = replace(estimator_config, baseline_hidden=tuple(mlp_profile["estimator_hidden"]))
                else:
                    estimator_config = replace(estimator_config, d_model=model_config.d_model,
                                               num_layers=model_config.num_layers, num_heads=model_config.num_heads,
                                               ffn_dim=model_config.ffn_dim)
                model = ActorCritic(estimator_config)
                encoder = parameter_count(model.actor.estimator)
                controller = parameter_count(model.actor.controller)
                total = encoder + controller
                assert total == parameter_count(model.actor) - model.actor.log_std.numel()
                if variant["latent_dim"]:
                    training_only = parameter_count(model.context_objective)
                basis = "instantiated_estimator_actor"
            expected = capacity["expected_deployment_parameters"][profile][variant["name"]]
            if total != expected:
                raise ValueError(f"{variant['name']}/{profile}: {total} != declared {expected}")
            limit = capacity["maximum_deployment_parameters"][profile]
            if total > limit:
                raise ValueError(f"{variant['name']}/{profile}: {total} exceeds {limit}")
            results.append({
                "variant": variant["name"], "profile": profile, "count_basis": basis,
                "encoder_parameters": encoder, "controller_parameters": controller,
                "deployment_parameters": total, "fp32_parameter_bytes": total * 4,
                "exploration_parameters_excluded": common["action_dim"],
                "auxiliary_training_parameters_excluded": training_only,
                "critic_parameters_excluded": parameter_count(ActorCritic(base).critic),
                "parameter_budget_passed": True,
            })
    screening = protocol["stages"]["screening"]
    compact = capacity["compact_ablation"]
    assert compact["training_seeds"] == screening["training_seeds"]
    assert compact["target_updates"] == screening["target_updates"]
    reports_per_checkpoint = (
        len(protocol["evaluation"]["validation_scenarios"])
        * len(protocol["evaluation"]["validation_seeds"])
    )
    compact_variants = compact["maximum_selected_families"] * len(compact["backbones_per_family"])
    compact_jobs = compact_variants * len(compact["training_seeds"])
    return {
        "protocol_revision": protocol["protocol_revision"], "models": results,
        "main_training_jobs": len(protocol["variants"]) * len(screening["training_seeds"]),
        "compact_additional_jobs_cap": compact_jobs,
        "compact_additional_training_transitions_cap": (
            compact_jobs * compact["target_updates"] * common["num_envs"] * common["rollout_steps"]
        ),
        "compact_additional_evaluation_reports_cap": (
            compact_jobs * len(screening["evaluation_checkpoint_updates"]) * reports_per_checkpoint
        ),
        "compact_wiring_training_transitions_cap": (
            compact_variants * compact["wiring_updates_per_variant"]
            * common["num_envs"] * common["rollout_steps"]
        ),
        "latency_verified_on_target": False, "new_estimator_policies_implemented": True,
        "scope": "CPU parameter inventory and budget validation only; no training or simulator",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    result = audit(args.protocol.resolve(strict=True))
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()

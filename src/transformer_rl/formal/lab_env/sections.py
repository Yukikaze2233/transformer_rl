"""Optional sections of the managed Lab runtime and their frozen contracts.

CPU-only and simulator-free, so the prepare-only entry, the managed runtime,
the diagnostic trainer and checkpoint evaluation parse sections identically.
Each section is parsed whole by its own config; the names match the
``build_env_cfg`` keywords. A reviewed timing preset may bind any optional
section and override fields of the runtime's sensors and termination limits.
"""
from __future__ import annotations

from ..control.actuation import MotorEnvelope
from ..control.communication import CommunicationConfig, EpisodeRandomizationConfig
from ..control.sensors import SensorModelConfig
from ..tasks.command_pool import CommandPoolConfig
from ..tasks.termination import NativeTaskLimits
from .action_transport import ActionLatchConfig
from .birth import BirthRandomizationConfig
from .materials import WheelMaterialConfig

OPTIONAL_SECTIONS = {'communication': CommunicationConfig, 'episode_randomization': EpisodeRandomizationConfig,
                     'wheel_material': WheelMaterialConfig, 'action_latch': ActionLatchConfig,
                     'motor_envelope': MotorEnvelope, 'command_pool': CommandPoolConfig,
                     'birth_randomization': BirthRandomizationConfig}
# A preset always freezes the communication budget and the episode randomization.
PRESET_REQUIRED = ('communication', 'episode_randomization')
_RETIRED_TRANSPORT = {'enabled': False, 'delays_steps': [0], 'nominal_fraction': 1.0}


def termination_limits(fields):
    """Runtime termination block (JSON lists) as the frozen limits."""
    return NativeTaskLimits(**{**fields, 'forbidden_body_names': tuple(fields['forbidden_body_names'])})


# Runtime blocks a preset overrides field by field; each result is re-parsed whole.
PRESET_OVERRIDES = {'sensors': lambda fields: SensorModelConfig(**fields), 'termination_limits': termination_limits}


def optional_sections(configuration):
    """Parsed optional sections as ``build_env_cfg`` keywords; None when absent."""
    return {name: None if configuration.get(name) is None else parser.from_dict(configuration[name])
            for name, parser in OPTIONAL_SECTIONS.items()}


def merge_preset(runtime, preset):
    """The runtime with a reviewed preset merged; the input is not mutated."""
    allowed = set(OPTIONAL_SECTIONS) | set(PRESET_OVERRIDES)
    if not isinstance(preset, dict) or not set(PRESET_REQUIRED) <= set(preset) <= allowed:
        raise ValueError('control preset must bind communication and episode randomization')
    merged = dict(preset)
    for name in set(preset) & set(OPTIONAL_SECTIONS):
        OPTIONAL_SECTIONS[name].from_dict(preset[name])
    for name, parser in PRESET_OVERRIDES.items():
        if name not in preset:
            continue
        # Overrides keep the runtime's other fields (e.g. clocks).
        if not isinstance(preset[name], dict) or not isinstance(runtime.get(name), dict):
            raise ValueError(f'{name} override requires a runtime {name} block')
        merged[name] = {**runtime[name], **preset[name]}
        parser(merged[name])
    value = dict(runtime)
    if 'action_latch' in preset and ActionLatchConfig.from_dict(preset['action_latch']).enabled:
        # The substep latch replaces whole policy-step transport.
        value['action_transport'] = dict(_RETIRED_TRANSPORT)
    value.update(merged)
    return value

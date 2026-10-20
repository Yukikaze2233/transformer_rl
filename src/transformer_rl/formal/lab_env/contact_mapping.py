"""Deterministic native-body to contact-sensor mapping; no simulator imports."""
from __future__ import annotations


def required_contact_bodies(native_body_names, wheel_body_indices, *, require_all=False):
    names = tuple(native_body_names)
    wheels = tuple(int(index) for index in wheel_body_indices)
    if (not names or len(set(names)) != len(names) or 'base_link' not in names
            or any(not isinstance(name, str) or not name for name in names)
            or len(wheels) != 2 or len(set(wheels)) != 2
            or any(index < 0 or index >= len(names) for index in wheels)
            or type(require_all) is not bool):
        raise ValueError("native contact topology requires base_link and two distinct wheel bodies")
    if require_all:
        return names
    required = {"base_link"} | {names[index] for index in wheels}
    return tuple(sorted(required))


def contact_body_mapping(native_body_names, sensor_body_names, wheel_body_indices, *, require_all=False):
    names = tuple(native_body_names)
    sensors = tuple(sensor_body_names)
    if len(set(sensors)) != len(sensors):
        raise RuntimeError("contact sensor body names are ambiguous")
    required = required_contact_bodies(names, wheel_body_indices, require_all=require_all)
    missing = tuple(name for name in required if name not in sensors)
    if missing:
        raise RuntimeError("contact sensor missing required native bodies: " + ", ".join(missing))
    source, target = [], []
    for index, name in enumerate(names):
        if name in sensors:
            target.append(index)
            source.append(sensors.index(name))
    if (not source or len(set(source)) != len(source) or len(set(target)) != len(target)):
        raise RuntimeError("contact sensor mapping is not one-to-one")
    return source, target, required

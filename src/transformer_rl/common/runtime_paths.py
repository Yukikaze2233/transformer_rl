"""Bind writable SDK paths to one owned worker without importing the SDK.

Requested paths and actual Kit readback are separate evidence. This module
does not measure runtime peaks or certify GPU-driver/helper-process behavior.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat


PROFILE_ENV = "TRANSFORMER_RL_RUNTIME_PROFILE"
PROFILE_SHA_ENV = "TRANSFORMER_RL_RUNTIME_PROFILE_SHA256"
PROFILE_FILENAME = "runtime.profile.json"
FORMAT = "transformer_rl.worker_runtime_paths"
REQUIRED_TOKENS = ("cache", "data", "logs", "omni_cache", "omni_global_data")
REQUIRED_SETTINGS = (
    "/app/portableMode", "/app/userConfigPath", "/app/extensions/registryEnabled",
    "/exts/omni.kit.registry.nucleus/cachePath",
    "/exts/omni.kit.registry.nucleus/cacheCreateLinks",
)
_ENV_PATH_KEYS = (
    "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME",
    "XDG_STATE_HOME", "WARP_CACHE_PATH", "TORCH_EXTENSIONS_DIR",
    "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH", "OMNICLIENT_HUB_CACHE_DIR",
)
RESERVED_ENVIRONMENT = frozenset((*_ENV_PATH_KEYS, "OMNICLIENT_HUB_MODE", PROFILE_ENV,
    PROFILE_SHA_ENV, "PYTHONPATH", "PYTHONPYCACHEPREFIX", "PYTHONDONTWRITEBYTECODE"))
_FIELDS = {"format", "schema_version", "worker_directory", "runtime_root", "paths",
           "environment", "tokens", "settings", "directory_identities"}


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _path(value):
    _require(isinstance(value, (str, Path)), "runtime path must be explicit")
    path = Path(value)
    _require(path.is_absolute() and str(path) == str(path.resolve()),
             "runtime path must be absolute, canonical and contain no symlink")
    # AppLauncher applies str.split(), rather than a shell/shlex parser.
    _require(not any(c.isspace() for c in str(path)),
             "AppLauncher runtime paths cannot contain whitespace")
    return path


def _directory_identity(path, device=None):
    path = _path(path)
    value = path.lstat()
    _require(stat.S_ISDIR(value.st_mode) and value.st_uid == os.getuid(),
             "runtime directory must be owned by this user")
    _require(device is None or value.st_dev == device,
             "runtime directory crosses the worker filesystem")
    return {"device": value.st_dev, "inode": value.st_ino, "uid": value.st_uid,
            "gid": value.st_gid, "mode": value.st_mode}


def _file_identity(value):
    # Reading a file may update atime; it is not an input-content mutation.
    return (value.st_dev, value.st_ino, value.st_uid, value.st_gid, value.st_mode,
            value.st_size, value.st_nlink, value.st_mtime_ns, value.st_ctime_ns)


def _layout(directory):
    root = directory / "runtime"
    relative = {
        "kit": "kit", "kit_cache": "kit/cache", "kit_data": "kit/data",
        "kit_logs": "kit/logs", "global_data": "kit/global_data",
        "global_cache": "kit/global_cache", "documents": "documents", "tmp": "tmp",
        "xdg": "xdg", "xdg_cache": "xdg/cache", "xdg_data": "xdg/data", "xdg_config": "xdg/config",
        "xdg_state": "xdg/state", "warp": "warp", "torch": "torch", "torch_extensions": "torch/extensions",
        "torch_inductor": "torch/inductor", "cuda": "cuda", "client": "client",
        "registry": "registry", "shader": "kit/cache/shadercache",
        "driver_shader": "kit/cache/nv_shadercache", "crash": "kit/data/crash",
    }
    paths = {name: str(root / value) for name, value in relative.items()}
    environment = {
        **{key: paths["tmp"] for key in ("TMPDIR", "TMP", "TEMP")},
        "XDG_CACHE_HOME": paths["xdg_cache"], "XDG_DATA_HOME": paths["xdg_data"],
        "XDG_CONFIG_HOME": paths["xdg_config"], "XDG_STATE_HOME": paths["xdg_state"],
        "WARP_CACHE_PATH": paths["warp"], "TORCH_EXTENSIONS_DIR": paths["torch_extensions"],
        "TORCHINDUCTOR_CACHE_DIR": paths["torch_inductor"], "CUDA_CACHE_PATH": paths["cuda"],
        "OMNICLIENT_HUB_MODE": "exclusive", "OMNICLIENT_HUB_CACHE_DIR": paths["client"],
    }
    tokens = {
        "cache": paths["kit_cache"], "data": paths["kit_data"], "logs": paths["kit_logs"],
        "omni_cache": paths["kit_cache"], "omni_data": paths["kit_data"],
        "omni_logs": paths["kit_logs"], "omni_global_data": paths["global_data"],
        "omni_global_cache": paths["global_cache"], "app_documents": paths["documents"],
        "shared_documents": paths["documents"], "temp": paths["tmp"],
    }
    settings = {
        "/app/portableMode": True, "/app/userConfigPath": paths["kit_data"] + "/user.config.json",
        "/app/extensions/registryEnabled": False,
        "/app/extensions/registryCache": paths["registry"],
        "/exts/omni.kit.registry.nucleus/cachePath": paths["registry"],
        "/exts/omni.kit.registry.nucleus/cacheCreateLinks": False,
        "/exts/omni.kit.telemetry/skipDeferredStartup": True,
        "/rtx/shaderDb/shaderCachePath": paths["shader"],
        "/rtx/shaderDb/driverShaderCachePath": paths["driver_shader"],
        "/crashreporter/dumpDir": paths["crash"], "/log/file": paths["kit_logs"] + "/kit.log",
    }
    return root, paths, environment, tokens, settings


def _validate_content(profile, directory):
    _require(type(profile) is dict and set(profile) == _FIELDS,
             "runtime profile schema differs")
    _require(profile["format"] == FORMAT and type(profile["schema_version"]) is int
             and profile["schema_version"] == 1, "runtime profile format differs")
    directory = _path(directory)
    root, paths, environment, tokens, settings = _layout(directory)
    expected = {"worker_directory": str(directory), "runtime_root": str(root),
                "paths": paths, "environment": environment, "tokens": tokens, "settings": settings}
    _require(all(_bytes(profile[key]) == _bytes(value) for key, value in expected.items()),
             "runtime profile paths or settings escape the declared worker")
    identities = profile["directory_identities"]
    identity_paths = {"worker_directory": str(directory), "runtime_root": str(root), **paths}
    _require(type(identities) is dict and set(identities) == set(identity_paths),
             "runtime directory identities differ")
    worker = _directory_identity(directory)
    for name, path in identity_paths.items():
        _require(_bytes(identities[name]) == _bytes(_directory_identity(path, worker["device"])),
                 "owned runtime directory identity changed")
    return profile


def prepare_worker_runtime(directory, environment):
    """Create a fresh private runtime tree; keep caller HOME and SDK inputs."""
    directory = _path(directory)
    worker = _directory_identity(directory)
    _require(type(environment) is dict and all(type(k) is str and type(v) is str
             for k, v in environment.items()), "worker environment must contain strings")
    root, paths, updates, tokens, settings = _layout(directory)
    profile_path = directory / PROFILE_FILENAME
    _require(not os.path.lexists(profile_path), "runtime profile cannot be overwritten")
    root.mkdir(mode=0o700)
    for value in paths.values():
        Path(value).mkdir(mode=0o700, parents=True, exist_ok=True)
    identity_paths = {"worker_directory": str(directory), "runtime_root": str(root), **paths}
    profile = {"format": FORMAT, "schema_version": 1, "worker_directory": str(directory),
        "runtime_root": str(root), "paths": paths, "environment": updates, "tokens": tokens,
        "settings": settings, "directory_identities": {
            key: _directory_identity(path, worker["device"]) for key, path in identity_paths.items()}}
    raw = _bytes(_validate_content(profile, directory))
    descriptor = os.open(profile_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    result = {**environment, **updates, PROFILE_ENV: str(profile_path),
              PROFILE_SHA_ENV: hashlib.sha256(raw).hexdigest()}
    validate_runtime_profile(directory, environ=result)
    return result


def _read_profile(path, expected_sha256, directory):
    path = _path(path)
    _require(path == directory / PROFILE_FILENAME, "runtime profile belongs to another worker")
    _require(type(expected_sha256) is str and len(expected_sha256) == 64
             and all(c in "0123456789abcdef" for c in expected_sha256),
             "external runtime profile SHA-256 is required")
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and before.st_nlink == 1
             and before.st_dev == directory.lstat().st_dev and before.st_size <= 65536,
             "runtime profile must be a small owned regular file on the worker filesystem")
    raw = path.read_bytes()
    _require(_file_identity(path.lstat()) == _file_identity(before)
             and hashlib.sha256(raw).hexdigest() == expected_sha256,
             "runtime profile changed or differs from its external SHA-256")
    profile = json.loads(raw)
    _require(raw == _bytes(profile), "runtime profile must use canonical JSON bytes")
    return _validate_content(profile, directory)


def validate_runtime_profile(directory=None, environ=None):
    """Validate inherited bindings against the immutable worker-request directory."""
    environment = os.environ if environ is None else environ
    _require(environment.get(PROFILE_ENV) and environment.get(PROFILE_SHA_ENV),
             "worker runtime profile and external SHA-256 are required")
    profile_path = _path(environment[PROFILE_ENV])
    directory = profile_path.parent if directory is None else _path(directory)
    profile = _read_profile(profile_path, environment[PROFILE_SHA_ENV], directory)
    _require(all(environment.get(key) == value for key, value in profile["environment"].items()),
             "worker runtime environment binding changed")
    return profile


def profile_receipt(profile):
    """Pin actual profile bytes for parent process and immutable result records."""
    directory = _path(profile["worker_directory"])
    path = directory / PROFILE_FILENAME
    raw = _bytes(profile)
    sha256 = hashlib.sha256(raw).hexdigest()
    actual = _read_profile(path, sha256, directory)
    _require(_bytes(actual) == raw, "runtime profile bytes changed")
    return {"path": str(path), "sha256": sha256, "bytes": len(raw)}


def validate_runtime_artifact(directory, receipt):
    """Validate a closed worker profile without inheriting that worker's environment."""
    directory = _path(directory)
    _require(type(receipt) is dict and set(receipt) == {"path", "sha256", "bytes"}
             and type(receipt["bytes"]) is int and receipt["bytes"] > 0,
             "worker runtime profile receipt differs")
    profile = _read_profile(receipt["path"], receipt["sha256"], directory)
    _require(profile_receipt(profile) == receipt, "worker runtime profile publication differs")
    return profile


def kit_arguments(profile):
    """Return literal whitespace-separated AppLauncher arguments, without shell quoting."""
    profile_receipt(profile)
    arguments = ["--portable-root", profile["paths"]["kit"]]
    arguments.extend(f"--/app/tokens/{key}={value}" for key, value in sorted(profile["tokens"].items()))
    for key, value in sorted(profile["settings"].items()):
        encoded = str(value).lower() if type(value) is bool else value
        arguments.append(f"--{key}={encoded}")
    return " ".join(arguments)


def _active_path(value, profile, *, directory=False):
    _require(type(value) is str and value and "${" not in value,
             "active runtime path is absent or unresolved")
    path = _path(value)
    root = _path(profile["runtime_root"])
    _require(path.is_relative_to(root), "active SDK output escapes the owned runtime")
    device = profile["directory_identities"]["runtime_root"]["device"]
    for component in (path, *path.parents):
        if not component.is_relative_to(root):
            break
        if not os.path.lexists(component):
            continue
        observed = component.lstat()
        _require(observed.st_uid == os.getuid() and observed.st_dev == device
                 and (stat.S_ISDIR(observed.st_mode) or (component == path and stat.S_ISREG(observed.st_mode))),
                 "active runtime path contains an unsafe entry")
    _require(not directory or path.is_dir(), "active runtime directory does not exist")
    return str(path)


def verify_kit_runtime(profile, token_resolver, settings_getter, experience_path):
    """Read actual Kit tokens/settings after startup, before creating the robot environment."""
    initial_profile = profile_receipt(profile)
    _require(callable(token_resolver) and callable(settings_getter), "actual Kit readback providers required")
    tokens = {}
    for name in profile["tokens"]:
        value = token_resolver("${" + name + "}")
        absent = value is None or value == "" or value == "${" + name + "}"
        if absent:
            _require(name not in REQUIRED_TOKENS, "required Kit runtime token is unresolved")
            tokens[name] = None
        else:
            tokens[name] = _active_path(value, profile, directory=True)
            _require(tokens[name] == profile["tokens"][name], "Kit runtime token differs from its worker binding")
    settings = {}
    for name, expected in profile["settings"].items():
        value = settings_getter(name)
        if value is None:
            _require(name not in REQUIRED_SETTINGS, "required Kit runtime setting is absent")
        elif type(expected) is bool:
            _require(type(value) is bool and value == expected, "Kit runtime Boolean setting differs")
        else:
            _active_path(value, profile)
            _require(value == expected, "active Kit output path differs from its worker binding")
        settings[name] = value
    experience = _path(experience_path)
    before = experience.lstat()
    _require(stat.S_ISREG(before.st_mode), "actual Kit experience must be a regular file")
    raw = experience.read_bytes()
    _require(_file_identity(experience.lstat()) == _file_identity(before),
             "actual Kit experience changed while reading")
    _require(profile_receipt(profile) == initial_profile, "runtime profile changed during Kit readback")
    return {"format": "transformer_rl.kit_runtime_readback", "schema_version": 1,
        "profile": initial_profile, "runtime_root": profile["runtime_root"],
        "portable_root": profile["paths"]["kit"], "tokens": tokens, "settings": settings,
        "experience": {"path": str(experience), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)},
        "hardware_verified": False}

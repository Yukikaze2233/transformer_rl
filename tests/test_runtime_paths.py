"""Owned filesystem checks for SDK runtime bindings; no SDK is imported.

The token/settings providers below model startup readback only.  They do not
represent an Isaac startup, a measured cache peak, or a system-wide quota.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pytest

from transformer_rl import runtime_paths as runtime


def raw_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


PATH_ENVIRONMENT_KEYS = (
    "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "XDG_DATA_HOME",
    "XDG_CONFIG_HOME", "XDG_STATE_HOME", "WARP_CACHE_PATH",
    "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR", "OMNICLIENT_HUB_CACHE_DIR",
    "CUDA_CACHE_PATH",
)


@pytest.fixture
def owned_runtime(tmp_path):
    worker = tmp_path / "worker"
    worker.mkdir()
    original = {"HOME": str(tmp_path / "unchanged_home"), "UNRELATED_KEY": "kept"}
    environment = runtime.prepare_worker_runtime(worker, original)
    profile = runtime.validate_runtime_profile(worker, environment)
    return {"worker": worker, "environment": environment, "profile": profile,
            "original": original,
            "profile_path": Path(environment[runtime.PROFILE_ENV])}


def rewrite_profile(fixture, value, *, raw=None):
    path = fixture["profile_path"]
    path.write_bytes(canonical_bytes(value) if raw is None else raw)
    environment = dict(fixture["environment"])
    environment[runtime.PROFILE_SHA_ENV] = raw_sha(path)
    return environment


def test_prepare_creates_owned_exclusive_root_and_external_raw_sha(owned_runtime):
    p = owned_runtime
    worker, environment = p["worker"], p["environment"]
    assert (worker / "runtime").is_dir()
    assert (worker / "runtime").stat().st_uid == os.getuid()
    assert p["profile_path"] == worker / "runtime.profile.json"
    assert p["profile_path"].stat().st_uid == os.getuid()
    assert environment[runtime.PROFILE_SHA_ENV] == raw_sha(p["profile_path"])
    assert p["original"] == {"HOME": str(worker.parent / "unchanged_home"),
                              "UNRELATED_KEY": "kept"}
    assert environment["HOME"] == p["original"]["HOME"]
    assert environment["UNRELATED_KEY"] == "kept"
    assert runtime.validate_runtime_profile(environ=environment) == p["profile"]


def test_worker_profiles_cannot_be_reused_or_cross_attached(owned_runtime):
    p = owned_runtime
    with pytest.raises((ValueError, FileExistsError)):
        runtime.prepare_worker_runtime(p["worker"], p["original"])
    sibling = p["worker"].parent / "second_worker"
    sibling.mkdir()
    second = runtime.prepare_worker_runtime(sibling, p["original"])
    assert second[runtime.PROFILE_ENV] != p["environment"][runtime.PROFILE_ENV]
    with pytest.raises(ValueError):
        runtime.validate_runtime_profile(sibling, p["environment"])


def test_inherited_shared_cache_paths_are_rebound_without_mutating_environment(tmp_path):
    worker = tmp_path / "worker"
    worker.mkdir()
    inherited = {key: str(tmp_path / "inherited_shared") for key in PATH_ENVIRONMENT_KEYS}
    inherited.update(HOME=str(tmp_path / "unchanged_home"), OMNICLIENT_HUB_MODE="shared")
    original = dict(inherited)
    environment = runtime.prepare_worker_runtime(worker, inherited)
    assert inherited == original
    assert environment["HOME"] == original["HOME"]
    assert environment["OMNICLIENT_HUB_MODE"] == "exclusive"
    root = worker / "runtime"
    for key in PATH_ENVIRONMENT_KEYS:
        path = Path(environment[key])
        assert path.is_absolute() and path.is_relative_to(root)
        assert path.is_dir() and not path.is_symlink()
        assert path.stat().st_uid == os.getuid()
        assert path.stat().st_dev == root.stat().st_dev
    runtime.validate_runtime_profile(worker, environment)


@pytest.mark.parametrize("key", PATH_ENVIRONMENT_KEYS)
@pytest.mark.parametrize("mutation", ["missing", "empty", "outside"])
def test_bound_environment_cannot_fall_back_to_shared_paths(owned_runtime, key, mutation):
    p = owned_runtime
    environment = dict(p["environment"])
    if mutation == "missing":
        del environment[key]
    else:
        environment[key] = "" if mutation == "empty" else str(p["worker"].parent)
    with pytest.raises(ValueError):
        runtime.validate_runtime_profile(p["worker"], environment)


@pytest.mark.parametrize("missing", ["profile", "sha"])
def test_external_profile_and_sha_are_both_required(owned_runtime, missing):
    environment = dict(owned_runtime["environment"])
    key = runtime.PROFILE_ENV if missing == "profile" else runtime.PROFILE_SHA_ENV
    del environment[key]
    with pytest.raises(ValueError):
        runtime.validate_runtime_profile(owned_runtime["worker"], environment)


def test_raw_byte_change_cannot_be_authorized_by_unchanged_sha(owned_runtime):
    path = owned_runtime["profile_path"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        runtime.validate_runtime_profile(owned_runtime["worker"],
                                         owned_runtime["environment"])


def test_noncanonical_and_unknown_profile_fields_fail_with_updated_sha(owned_runtime):
    p = owned_runtime
    value = json.loads(p["profile_path"].read_bytes())
    environment = rewrite_profile(p, value, raw=json.dumps(value, indent=2).encode())
    with pytest.raises(ValueError):
        runtime.validate_runtime_profile(p["worker"], environment)
    value["undeclared_mutable_runtime"] = str(p["worker"].parent)
    environment = rewrite_profile(p, value)
    with pytest.raises(ValueError, match="schema"):
        runtime.validate_runtime_profile(p["worker"], environment)


@pytest.mark.parametrize("field,value", [("format", "different.runtime.profile"),
                                         ("schema_version", 2), ("schema_version", "1")])
def test_profile_schema_changes_fail_despite_matching_external_sha(owned_runtime, field, value):
    p = owned_runtime
    changed = deepcopy(p["profile"])
    changed[field] = value
    environment = rewrite_profile(p, changed)
    with pytest.raises(ValueError):
        runtime.validate_runtime_profile(p["worker"], environment)


@pytest.mark.parametrize("field", ["worker_directory", "runtime_root"])
@pytest.mark.parametrize("value", ["", "relative_runtime", "/"])
def test_declared_roots_are_absolute_owned_fixed_paths(owned_runtime, field, value):
    p = owned_runtime
    changed = deepcopy(p["profile"])
    changed[field] = value
    environment = rewrite_profile(p, changed)
    with pytest.raises(ValueError):
        runtime.validate_runtime_profile(p["worker"], environment)


def test_forged_sha_and_relative_profile_path_cannot_use_fallbacks(owned_runtime, monkeypatch):
    p = owned_runtime
    environment = dict(p["environment"])
    environment[runtime.PROFILE_SHA_ENV] = "0" * 64
    with pytest.raises(ValueError):
        runtime.validate_runtime_profile(p["worker"], environment)
    monkeypatch.chdir(p["worker"])
    environment = dict(p["environment"])
    environment[runtime.PROFILE_ENV] = "runtime.profile.json"
    with pytest.raises(ValueError):
        runtime.validate_runtime_profile(p["worker"], environment)


def test_profile_file_cannot_be_a_symlink_even_with_identical_bytes(owned_runtime):
    p = owned_runtime
    retained = p["worker"].parent / "retained_profile.json"
    p["profile_path"].rename(retained)
    p["profile_path"].symlink_to(retained)
    assert raw_sha(p["profile_path"]) == p["environment"][runtime.PROFILE_SHA_ENV]
    with pytest.raises(ValueError):
        runtime.validate_runtime_profile(p["worker"], p["environment"])


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_profile_publication_must_recheck_actual_file_type_and_links(owned_runtime, alias):
    p = owned_runtime
    retained = p["worker"].parent / "retained_profile.json"
    if alias == "symlink":
        p["profile_path"].rename(retained)
        p["profile_path"].symlink_to(retained)
    else:
        retained.hardlink_to(p["profile_path"])
    assert raw_sha(p["profile_path"]) == p["environment"][runtime.PROFILE_SHA_ENV]
    with pytest.raises(ValueError):
        runtime.profile_receipt(p["profile"])


@pytest.mark.parametrize("replacement", ["new_directory", "symlink"])
def test_runtime_root_replacement_is_rejected_by_identity(owned_runtime, replacement):
    p = owned_runtime
    root = p["worker"] / "runtime"
    retained = p["worker"].parent / "retained_runtime"
    root.rename(retained)
    if replacement == "symlink":
        root.symlink_to(retained, target_is_directory=True)
    else:
        root.mkdir()
    with pytest.raises(ValueError):
        runtime.validate_runtime_profile(p["worker"], p["environment"])


@pytest.mark.parametrize("field", ["device", "inode", "uid", "gid", "mode"])
def test_directory_metadata_cannot_be_resealed_to_different_identity(owned_runtime, field):
    p = owned_runtime
    changed = deepcopy(p["profile"])
    changed["directory_identities"]["runtime_root"][field] += 1
    environment = rewrite_profile(p, changed)
    with pytest.raises(ValueError, match="identity"):
        runtime.validate_runtime_profile(p["worker"], environment)


@pytest.mark.parametrize("field", ["paths", "environment", "tokens", "settings", "directory_identities"])
def test_nested_schema_has_no_undeclared_keys(owned_runtime, field):
    p = owned_runtime
    changed = deepcopy(p["profile"])
    changed[field]["unknown_binding"] = str(p["worker"].parent)
    environment = rewrite_profile(p, changed)
    with pytest.raises(ValueError):
        runtime.validate_runtime_profile(p["worker"], environment)


def test_actual_bound_subdirectory_inode_replacement_is_rejected(owned_runtime):
    p = owned_runtime
    path = Path(p["profile"]["paths"]["tmp"])
    retained = p["worker"] / "retained_tmp"
    path.rename(retained)
    path.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="identity"):
        runtime.validate_runtime_profile(p["worker"], p["environment"])


@pytest.mark.parametrize("parent_key,leaf_keys", [
    ("xdg", ("xdg_cache", "xdg_data", "xdg_config", "xdg_state")),
    ("torch", ("torch_extensions", "torch_inductor")),
])
def test_ancestor_replacement_is_rejected_even_when_leaf_inodes_are_preserved(
        owned_runtime, parent_key, leaf_keys):
    p = owned_runtime
    ancestor = Path(p["profile"]["paths"][parent_key])
    retained = p["worker"] / f"retained_{parent_key}"
    ancestor.rename(retained)
    ancestor.mkdir(mode=0o700)
    for leaf in retained.iterdir():
        leaf.rename(ancestor / leaf.name)
    for name in leaf_keys:
        assert Path(p["profile"]["paths"][name]).stat().st_ino == (
            p["profile"]["directory_identities"][name]["inode"])
    assert ancestor.stat().st_ino != p["profile"]["directory_identities"][parent_key]["inode"]
    with pytest.raises(ValueError, match="identity"):
        runtime.validate_runtime_profile(p["worker"], p["environment"])


def test_normal_cache_writes_and_new_files_do_not_change_directory_identity(owned_runtime):
    p = owned_runtime
    paths = p["profile"]["paths"]
    (Path(paths["kit_cache"]) / "fast_importer.pickle").write_bytes(b"owned cache fixture")
    (Path(paths["kit_logs"]) / "kit.log").write_text("owned log fixture\n")
    (Path(paths["warp"]) / "versioned_cache").mkdir()
    assert runtime.validate_runtime_profile(p["worker"], p["environment"]) == p["profile"]


def test_closed_worker_profile_receipt_is_validated_without_inherited_environment(owned_runtime):
    p = owned_runtime
    receipt = {"path": str(p["profile_path"]), "sha256": raw_sha(p["profile_path"]),
               "bytes": p["profile_path"].stat().st_size}
    assert runtime.validate_runtime_artifact(p["worker"], receipt) == p["profile"]
    changed = {**receipt, "bytes": receipt["bytes"] + 1}
    with pytest.raises(ValueError):
        runtime.validate_runtime_artifact(p["worker"], changed)
    changed = {**receipt, "sha256": "0" * 64}
    with pytest.raises(ValueError):
        runtime.validate_runtime_artifact(p["worker"], changed)


def test_actual_foreign_filesystem_directory_is_rejected(owned_runtime):
    p = owned_runtime
    alternate = Path("/dev/shm")
    if not alternate.is_dir() or not os.access(alternate, os.W_OK):
        pytest.skip("No writable alternate filesystem for this OS-only check")
    device = p["worker"].stat().st_dev
    if alternate.stat().st_dev == device:
        pytest.skip("Alternate fixture filesystem matches worker filesystem")
    # Only this exclusively created directory is removed by the context;
    # nothing else under the shared mount is enumerated or cleaned.
    with tempfile.TemporaryDirectory(prefix="transformer-runtime-crossfs-", dir=alternate) as base:
        path = Path(base)
        assert path.stat().st_uid == os.getuid() and path.stat().st_dev != device
        with pytest.raises(ValueError, match="filesystem"):
            runtime._directory_identity(path, device)


def kit_settings(profile):
    """Read the public CLI contract independently of the profile's schema."""
    values = {}
    for argument in runtime.kit_arguments(profile).split():
        if not argument.startswith("--/") or "=" not in argument:
            continue
        key, value = argument[2:].split("=", 1)
        values[key] = True if value == "true" else False if value == "false" else value
    return values


def readback_fixture(p):
    settings = kit_settings(p["profile"])
    settings["/app/portableMode"] = True
    tokens = {key.removeprefix("/app/tokens/"): value
              for key, value in settings.items() if key.startswith("/app/tokens/")}
    experience = p["worker"].parent / "sdk_input" / "headless.kit"
    experience.parent.mkdir()
    experience.write_bytes(b'[settings]\nfixture = "readback-only"\n')
    return tokens, settings, experience


def resolver(tokens):
    def resolve(expression):
        assert expression.startswith("${") and expression.endswith("}")
        return tokens.get(expression[2:-1], expression)
    return resolve


def test_kit_cli_has_two_token_portable_root_and_explicit_mutable_bindings(owned_runtime):
    p = owned_runtime
    arguments = runtime.kit_arguments(p["profile"]).split()
    assert arguments.count("--portable-root") == 1
    index = arguments.index("--portable-root")
    portable = Path(arguments[index + 1])
    assert portable.is_absolute() and portable.is_relative_to(p["worker"] / "runtime")
    assert portable.is_dir() and not portable.is_symlink()
    settings = kit_settings(p["profile"])
    for name in ("cache", "data", "logs", "omni_cache", "omni_global_data"):
        bound = Path(settings[f"/app/tokens/{name}"])
        assert bound.is_absolute() and bound.is_relative_to(p["worker"] / "runtime")
        assert bound.is_dir() and bound.stat().st_uid == os.getuid()
    assert settings["/app/extensions/registryEnabled"] is False
    assert settings["/exts/omni.kit.registry.nucleus/cacheCreateLinks"] is False
    assert Path(settings["/exts/omni.kit.registry.nucleus/cachePath"]).is_relative_to(
        p["worker"] / "runtime")
    assert Path(settings["/app/userConfigPath"]).is_relative_to(p["worker"] / "runtime")


@pytest.mark.parametrize("name", ["cache", "data", "logs", "omni_cache", "omni_global_data"])
@pytest.mark.parametrize("mutation", ["missing", "empty", "outside", "relative"])
def test_required_resolved_tokens_cannot_be_missing_or_escape(owned_runtime, name, mutation):
    p = owned_runtime
    tokens, settings, experience = readback_fixture(p)
    if mutation == "missing":
        del tokens[name]
    else:
        tokens[name] = {"empty": "", "outside": str(p["worker"].parent),
                        "relative": "relative_cache"}[mutation]
    with pytest.raises(ValueError):
        runtime.verify_kit_runtime(p["profile"], resolver(tokens), settings.get, experience)


@pytest.mark.parametrize("key", ["/app/portableMode", "/app/userConfigPath",
    "/app/extensions/registryEnabled", "/exts/omni.kit.registry.nucleus/cachePath",
    "/exts/omni.kit.registry.nucleus/cacheCreateLinks"])
def test_mandatory_runtime_settings_must_have_actual_readback(owned_runtime, key):
    p = owned_runtime
    tokens, settings, experience = readback_fixture(p)
    del settings[key]
    with pytest.raises(ValueError):
        runtime.verify_kit_runtime(p["profile"], resolver(tokens), settings.get, experience)


@pytest.mark.parametrize("key", ["/rtx/shaderDb/shaderCachePath",
                                 "/rtx/shaderDb/driverShaderCachePath"])
def test_active_optional_shader_cache_cannot_use_shared_sdk_paths(owned_runtime, key):
    p = owned_runtime
    tokens, settings, experience = readback_fixture(p)
    settings[key] = str(experience.parent)
    with pytest.raises(ValueError):
        runtime.verify_kit_runtime(p["profile"], resolver(tokens), settings.get, experience)


def test_resolved_descendant_symlink_cannot_escape_to_same_owner(owned_runtime):
    p = owned_runtime
    tokens, settings, experience = readback_fixture(p)
    path = p["worker"] / "runtime" / "linked_shared_cache"
    path.symlink_to(experience.parent, target_is_directory=True)
    tokens["cache"] = str(path)
    with pytest.raises(ValueError):
        runtime.verify_kit_runtime(p["profile"], resolver(tokens), settings.get, experience)


def test_actual_readback_records_experience_raw_bytes_and_unverified_hardware(owned_runtime):
    p = owned_runtime
    tokens, settings, experience = readback_fixture(p)
    # Packaged app shader inputs are immutable SDK dependencies, not writable
    # worker caches.  Their outside paths must not be mistaken for an escape.
    settings["/rtx/shaderDb/appShaderCachePath"] = str(experience.parent)
    settings["/rtx/shaderDb/driverAppShaderCachePath"] = str(experience.parent)
    result = runtime.verify_kit_runtime(p["profile"], resolver(tokens), settings.get, experience)
    assert result["experience"] == {"path": str(experience), "sha256": raw_sha(experience),
                                    "bytes": experience.stat().st_size}
    assert result["profile"] == {"path": str(p["profile_path"]),
                                 "sha256": raw_sha(p["profile_path"]),
                                 "bytes": p["profile_path"].stat().st_size}
    assert result["runtime_root"] == str(p["worker"] / "runtime")
    assert Path(result["portable_root"]).is_relative_to(p["worker"] / "runtime")
    for name in ("cache", "data", "logs", "omni_cache", "omni_global_data"):
        assert result["tokens"][name] == tokens[name]
    for key in ("/app/portableMode", "/app/userConfigPath", "/app/extensions/registryEnabled",
                "/exts/omni.kit.registry.nucleus/cachePath",
                "/exts/omni.kit.registry.nucleus/cacheCreateLinks"):
        assert result["settings"][key] == settings[key]
    assert result["hardware_verified"] is False


@pytest.mark.parametrize("read_target", ["profile", "experience"])
def test_actual_read_access_time_change_is_not_an_input_mutation(owned_runtime, read_target):
    p = owned_runtime
    tokens, settings, experience = readback_fixture(p)
    path = p["profile_path"] if read_target == "profile" else experience
    observed = path.stat()
    os.utime(path, ns=(0, observed.st_mtime_ns))
    before = path.stat()
    assert before.st_atime_ns == 0
    result = runtime.verify_kit_runtime(p["profile"], resolver(tokens), settings.get, experience)
    after = path.stat()
    if after.st_atime_ns == before.st_atime_ns:
        pytest.skip("Fixture filesystem does not update atime on actual reads")
    assert (after.st_dev, after.st_ino, after.st_uid, after.st_gid, after.st_mode,
            after.st_size, after.st_nlink, after.st_mtime_ns, after.st_ctime_ns) == (
        before.st_dev, before.st_ino, before.st_uid, before.st_gid, before.st_mode,
        before.st_size, before.st_nlink, before.st_mtime_ns, before.st_ctime_ns)
    assert result[read_target]["sha256"] == raw_sha(path)


def test_inactive_optional_tokens_and_settings_need_not_exist(owned_runtime):
    p = owned_runtime
    tokens, settings, experience = readback_fixture(p)
    required_tokens = {name: tokens[name] for name in runtime.REQUIRED_TOKENS}
    required_settings = {name: settings[name] for name in runtime.REQUIRED_SETTINGS}
    result = runtime.verify_kit_runtime(p["profile"], resolver(required_tokens),
                                        required_settings.get, experience)
    assert all(value is None for name, value in result["tokens"].items()
               if name not in runtime.REQUIRED_TOKENS)
    assert all(value is None for name, value in result["settings"].items()
               if name not in runtime.REQUIRED_SETTINGS)


def test_kit_readback_cannot_publish_profile_changed_to_symlink(owned_runtime):
    p = owned_runtime
    tokens, settings, experience = readback_fixture(p)
    retained = p["worker"].parent / "retained_profile.json"
    p["profile_path"].rename(retained)
    p["profile_path"].symlink_to(retained)
    with pytest.raises(ValueError):
        runtime.verify_kit_runtime(p["profile"], resolver(tokens), settings.get, experience)


@pytest.mark.parametrize("mutation", ["symlink", "bytes"])
def test_profile_is_rechecked_after_token_readback_side_effect(owned_runtime, mutation):
    p = owned_runtime
    tokens, settings, experience = readback_fixture(p)
    original_resolver = resolver(tokens)
    changed = False

    def resolve_then_mutate(expression):
        nonlocal changed
        if not changed:
            if mutation == "symlink":
                retained = p["worker"].parent / "retained_profile.json"
                p["profile_path"].rename(retained)
                p["profile_path"].symlink_to(retained)
            else:
                p["profile_path"].write_bytes(p["profile_path"].read_bytes() + b" ")
            changed = True
        return original_resolver(expression)

    with pytest.raises(ValueError):
        runtime.verify_kit_runtime(p["profile"], resolve_then_mutate, settings.get, experience)
    assert changed, "The initial pin must pass before this mid-readback mutation"


@pytest.mark.parametrize("name", ["omni_data", "omni_logs", "omni_global_cache"])
def test_active_optional_tokens_must_also_be_worker_owned(owned_runtime, name):
    p = owned_runtime
    tokens, settings, experience = readback_fixture(p)
    tokens[name] = str(experience.parent)
    with pytest.raises(ValueError):
        runtime.verify_kit_runtime(p["profile"], resolver(tokens), settings.get, experience)


@pytest.mark.parametrize("key,value", [("/app/portableMode", False),
    ("/app/extensions/registryEnabled", True),
    ("/exts/omni.kit.registry.nucleus/cacheCreateLinks", True)])
def test_runtime_mode_and_registry_actual_values_cannot_differ(owned_runtime, key, value):
    p = owned_runtime
    tokens, settings, experience = readback_fixture(p)
    settings[key] = value
    with pytest.raises(ValueError):
        runtime.verify_kit_runtime(p["profile"], resolver(tokens), settings.get, experience)


@pytest.mark.parametrize("key", ["/app/userConfigPath", "/exts/omni.kit.registry.nucleus/cachePath"])
def test_runtime_settings_cannot_escape_with_same_owner_and_filesystem(owned_runtime, key):
    p = owned_runtime
    tokens, settings, experience = readback_fixture(p)
    settings[key] = str(experience.parent)
    with pytest.raises(ValueError):
        runtime.verify_kit_runtime(p["profile"], resolver(tokens), settings.get, experience)


@pytest.mark.parametrize("name", ["worker with_space", "worker\twith_tab", "worker\nwith_newline"])
def test_whitespace_worker_paths_are_rejected_before_any_runtime_creation(tmp_path, name):
    worker = tmp_path / name
    worker.mkdir()
    with pytest.raises(ValueError):
        runtime.prepare_worker_runtime(worker, {})
    assert not (worker / "runtime").exists()
    assert not (worker / "runtime.profile.json").exists()


def test_worker_directory_itself_cannot_be_a_symlink(tmp_path):
    actual = tmp_path / "actual_worker"
    actual.mkdir()
    linked = tmp_path / "linked_worker"
    linked.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError):
        runtime.prepare_worker_runtime(linked, {})
    assert not (actual / "runtime").exists()

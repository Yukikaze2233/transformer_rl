"""Recovery boundaries must tolerate append/rename without false hash failures."""
import argparse
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


SPEC = importlib.util.spec_from_file_location(
    "recover_study", Path(__file__).parents[1] / "tools/recover_study.py")
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def test_append_and_atomic_replace_use_opened_prefix(tmp_path, monkeypatch):
    study = tmp_path / "study"
    study.mkdir()
    metrics = study / "metrics.jsonl"
    metrics.write_bytes(b'{"update":1}\n{"upd')
    status = study / "status.json"
    status.write_text('{"version":1}')
    original = recovery.PrefixReader.read
    mutated = set()

    def mutate_then_read(reader, size):
        # fdopen's name is an fd; target the held inode via /proc on Linux.
        name = Path(f"/proc/self/fd/{reader.stream.fileno()}").resolve().name
        if name not in mutated:
            mutated.add(name)
            if name == "metrics.jsonl":
                with metrics.open("ab") as stream:
                    stream.write(b'ate":2}\n')
            elif name == "status.json":
                replacement = study / "replacement"
                replacement.write_text('{"version":2}')
                replacement.replace(status)
        return original(reader, size)

    monkeypatch.setattr(recovery.PrefixReader, "read", mutate_then_read)
    output = tmp_path / "snapshot"
    recovery.snapshot(argparse.Namespace(study=study, output=output, include=[], git_source=[]))
    recovery.verify(argparse.Namespace(root=output))
    assert (output / "extracted/study/metrics.jsonl").read_bytes() == b'{"update":1}\n{"upd'
    assert json.loads((output / "extracted/study/status.json").read_text()) == {"version": 1}
    manifest = json.loads((output / "manifest.json").read_text())
    entry = next(e for e in manifest["files"] if e["path"].endswith("metrics.jsonl"))
    assert entry["jsonl"]["partial_tail_bytes"] == 5
    assert manifest["status_end"]["value"] == {"version": 2}
    with pytest.raises(FileExistsError):
        recovery.verify(argparse.Namespace(root=output))


def test_shortened_source_fails():
    reader = recovery.PrefixReader(io.BytesIO(b"abc"), 4)
    assert reader.read(4) == b"abc"
    with pytest.raises(EOFError):
        reader.read(1)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "a//b", "a\\b"])
def test_reject_unsafe_names(name):
    with pytest.raises(ValueError):
        recovery.safe_name(name)


@pytest.mark.parametrize("kind", ["link", "duplicate", "tampered"])
def test_verify_rejects_bad_archive_before_extraction(tmp_path, kind):
    archive_path = tmp_path / "snapshot.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("study/file")
        if kind == "link":
            member.type = tarfile.SYMTYPE
            member.linkname = "../../escape"
        archive.addfile(member)
        if kind == "duplicate":
            archive.addfile(member)
    manifest = {"archive": {"path": archive_path.name, "size": archive_path.stat().st_size,
                            "sha256": recovery.sha256(archive_path)},
                "files": [{"path": "study/file", "size": 0}], "file_count": 1, "total_bytes": 0}
    if kind == "tampered":
        manifest["archive"]["sha256"] = "0" * 64
    recovery.write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError):
        recovery.verify(argparse.Namespace(root=tmp_path))
    assert not (tmp_path / "extracted").exists()

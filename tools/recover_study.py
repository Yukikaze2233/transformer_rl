"""Read-only, bounded-prefix study snapshots and verified local extraction.

Only append-only files, atomically replaced files and immutable published files
are supported. Each file has its own open/fstat boundary, not a global instant.
The output directory must be new and outside every input tree.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tarfile
import traceback


CHUNK = 1024 * 1024


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts
            or str(path) != name or "\\" in name):
        raise ValueError(f"unsafe or noncanonical archive path: {name!r}")
    return path


class PrefixReader:
    """Hash exactly the bytes consumed by tar, never the later source state."""

    def __init__(self, stream, size):
        self.stream = stream
        self.remaining = size
        self.digest = hashlib.sha256()
        self.newlines = 0
        self.tail_bytes = 0

    def read(self, size):
        data = self.stream.read(min(size, self.remaining))
        if not data and self.remaining:
            raise EOFError("source shortened after fstat")
        self.remaining -= len(data)
        self.digest.update(data)
        self.newlines += data.count(b"\n")
        last = data.rfind(b"\n")
        self.tail_bytes = len(data) - last - 1 if last >= 0 else self.tail_bytes + len(data)
        return data


def read_status(study):
    # read_bytes opens once; atomic rename cannot change the opened inode.
    path = study / "status.json"
    if not path.exists():
        return None
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        data = stream.read(size)
    if len(data) != size:
        raise EOFError("status shortened after fstat")
    return {"observed_at": timestamp(), "sha256": hashlib.sha256(data).hexdigest(),
            "value": json.loads(data)}


def snapshot(args):
    study = args.study.resolve(strict=True)
    inputs = [("study", study, False)]
    for spec, tracked in [(s, False) for s in args.include] + [
        (s, True) for s in args.git_source
    ]:
        label, path = spec.split("=", 1)
        safe_name(label)
        inputs.append((label, Path(path).resolve(strict=True), tracked))
    output = args.output.resolve()
    for _, path, _ in inputs:
        if output == path or output.is_relative_to(path):
            raise ValueError("snapshot output must be outside every input")
    output.mkdir(parents=True, exist_ok=False)
    start = timestamp()
    try:
        status_start = read_status(study)
        inventory, sources, skipped = {}, [], []
        for label, root, tracked in inputs:
            source = {"label": label, "path": str(root), "git_tracked_only": tracked}
            if tracked:
                def git(*command):
                    return subprocess.check_output(
                        ["git", "-C", str(root), *command], timeout=30)
                source["commit"] = git("rev-parse", "HEAD").decode().strip()
                source["status_porcelain"] = git("status", "--porcelain").decode()
                paths = [root / os.fsdecode(p) for p in git("ls-files", "-z").split(b"\0") if p]
            else:
                paths = sorted(root.rglob("*")) if root.is_dir() else [root]
            sources.append(source)
            for path in paths:
                if path.is_symlink():
                    raise ValueError(f"symlink input requires explicit resolution: {path}")
                if path.is_dir():
                    continue
                # Atomic publication temporary files are not published artifacts.
                if path.name.startswith(".") and path.name.endswith(".tmp"):
                    skipped.append({"path": str(path), "reason": "unpublished temporary"})
                    continue
                name = label + ("/" + path.relative_to(root).as_posix() if root.is_dir() else "")
                safe_name(name)
                if name in inventory:
                    raise ValueError(f"duplicate input: {name}")
                inventory[name] = path
        entries = []
        archive_path = output / "snapshot.tar.gz"
        with archive_path.open("xb") as archive_file, gzip.GzipFile(
            fileobj=archive_file, mode="wb", compresslevel=1
        ) as compressed, tarfile.open(fileobj=compressed, mode="w|") as archive:
            for name, path in sorted(inventory.items()):
                opened_at = timestamp()
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(fd, "rb") as stream:
                    before = os.fstat(stream.fileno())
                    if not stat.S_ISREG(before.st_mode):
                        raise ValueError(f"not a regular file: {path}")
                    info = tarfile.TarInfo(name)
                    info.size = before.st_size
                    info.mode = stat.S_IMODE(before.st_mode) & 0o777
                    info.mtime = before.st_mtime
                    reader = PrefixReader(stream, before.st_size)
                    archive.addfile(info, reader)
                    if reader.remaining:
                        raise EOFError(f"incomplete tar member: {name}")
                    after = os.fstat(stream.fileno())
                entry = {
                    "path": name, "source": str(path), "size": before.st_size,
                    "sha256": reader.digest.hexdigest(), "opened_at": opened_at,
                    "copied_at": timestamp(), "device": before.st_dev, "inode": before.st_ino,
                    "mtime_ns_at_open": before.st_mtime_ns, "size_after_read": after.st_size,
                    "boundary": "opened_fd_fstat_size_prefix",
                }
                if path.suffix == ".jsonl":
                    entry["jsonl"] = {"newline_count": reader.newlines,
                                      "partial_tail_bytes": reader.tail_bytes,
                                      "partial_tail_preserved": bool(reader.tail_bytes)}
                entries.append(entry)
        end = timestamp()
        manifest = {
            "schema_version": 1, "snapshot_start": start, "snapshot_end": end,
            "semantics": "per-file open/fstat bounded prefixes; not a global atomic snapshot",
            "study": str(study), "sources": sources, "skipped": skipped,
            "status_start": status_start, "status_end": read_status(study),
            "inventory_files": len(inventory), "file_count": len(entries),
            "total_bytes": sum(e["size"] for e in entries), "files": entries,
            "archive": {"path": archive_path.name, "size": archive_path.stat().st_size,
                        "sha256": sha256(archive_path)},
        }
        write_json(output / "manifest.json", manifest)
        write_json(output / "snapshot-exit.json", {"exit_code": 0, "at": timestamp()})
        print(json.dumps({k: manifest[k] for k in (
            "snapshot_start", "snapshot_end", "file_count", "total_bytes", "archive")}))
    except BaseException:
        write_json(output / "snapshot-exit.json", {
            "exit_code": 1, "at": timestamp(), "traceback": traceback.format_exc()})
        raise


def verify(args):
    root = args.root.resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    archive_meta = manifest["archive"]
    archive_path = root / str(safe_name(archive_meta["path"]))
    if archive_path.stat().st_size != archive_meta["size"] or sha256(archive_path) != archive_meta["sha256"]:
        raise ValueError("archive size/SHA-256 mismatch")
    entries = {}
    for entry in manifest["files"]:
        name = str(safe_name(entry["path"]))
        if name in entries:
            raise ValueError(f"duplicate manifest path: {name}")
        entries[name] = entry
    if len(entries) != manifest["file_count"] or sum(e["size"] for e in entries.values()) != manifest["total_bytes"]:
        raise ValueError("manifest totals mismatch")
    # Validate every member before creating any extracted file. Links and special
    # files are forbidden; no extractall, permissions or ownership are trusted.
    seen = set()
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            name = str(safe_name(member.name))
            if not member.isfile() or name in seen or name not in entries:
                raise ValueError(f"unexpected archive member: {name}")
            if member.size != entries[name]["size"]:
                raise ValueError(f"member size mismatch: {name}")
            seen.add(name)
    if seen != entries.keys():
        raise ValueError("archive member set differs from manifest")
    destination = root / "extracted"
    destination.mkdir(exist_ok=False)
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            path = destination / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with archive.extractfile(member) as source, path.open("xb") as target:
                for block in iter(lambda: source.read(CHUNK), b""):
                    target.write(block)
                    digest.update(block)
            if digest.hexdigest() != entries[member.name]["sha256"]:
                raise ValueError(f"extracted SHA-256 mismatch: {member.name}")
    for name, entry in entries.items():
        path = destination / name
        if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"on-disk verification failed: {name}")
    receipt = {
        "schema_version": 1, "verified_at": timestamp(), "local_root": str(root),
        "snapshot_start": manifest["snapshot_start"], "snapshot_end": manifest["snapshot_end"],
        "remote_study": manifest["study"], "source_status": manifest["sources"],
        "status_start": manifest["status_start"], "status_end": manifest["status_end"],
        "file_count": len(entries), "total_bytes": manifest["total_bytes"],
        "archive": archive_meta, "archive_sha256_verified": True,
        "archive_paths_safe": True, "all_file_sha256_verified": True,
        "on_disk_recheck_count": len(entries), "manifest_sha256": sha256(root / "manifest.json"),
        "file_hashes_and_boundaries": "manifest.json",
        "partial_jsonl_tails": [e["path"] for e in entries.values()
                               if e.get("jsonl", {}).get("partial_tail_preserved")],
    }
    write_json(root / "recovery_receipt.json", receipt)
    print(json.dumps({"verified_files": len(entries), "bytes": manifest["total_bytes"],
                      "receipt": str(root / "recovery_receipt.json")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("snapshot")
    collect.add_argument("--study", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--include", action="append", default=[], metavar="LABEL=PATH")
    collect.add_argument("--git-source", action="append", default=[], metavar="LABEL=PATH")
    collect.set_defaults(function=snapshot)
    check = commands.add_parser("verify")
    check.add_argument("--root", type=Path, required=True)
    check.set_defaults(function=verify)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

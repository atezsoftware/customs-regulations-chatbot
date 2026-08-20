#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

_ATTESTATION_MAX_BYTES = 64 * 1024
_EVIDENCE_MAX_BYTES = 4 * 1024 * 1024
_ATTESTATION_SNAPSHOT = "regulatory-capabilities.json"
_EVIDENCE_SNAPSHOT = "regulatory-capability-evidence.json"
_READINESS_UID = 1001
_READINESS_GID = 1001


class SnapshotError(RuntimeError):
    pass


@dataclass(frozen=True)
class _SecureBytes:
    contents: bytes
    device: int
    inode: int


def _required_flag(name: str) -> int:
    value = getattr(os, name, None)
    if not isinstance(value, int) or value == 0:
        raise SnapshotError("required secure-open semantics are unavailable")
    return value


def _read_source(
    path: Path,
    *,
    expected_mode: int,
    expected_owner_uid: int,
    expected_owner_gid: int,
    maximum_size: int,
) -> _SecureBytes:
    flags = (
        os.O_RDONLY
        | _required_flag("O_NONBLOCK")
        | _required_flag("O_NOFOLLOW")
        | _required_flag("O_CLOEXEC")
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SnapshotError("source open failed") from error

    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SnapshotError("source is not regular")
        if (
            stat.S_IMODE(metadata.st_mode) != expected_mode
            or metadata.st_uid != expected_owner_uid
            or metadata.st_gid != expected_owner_gid
        ):
            raise SnapshotError("source ownership or mode is invalid")
        if metadata.st_size <= 0 or metadata.st_size > maximum_size:
            raise SnapshotError("source size is invalid")

        contents = bytearray()
        while len(contents) <= maximum_size:
            remaining = maximum_size + 1 - len(contents)
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            contents.extend(chunk)
        if not contents or len(contents) > maximum_size:
            raise SnapshotError("source size is invalid")
        return _SecureBytes(
            contents=bytes(contents),
            device=metadata.st_dev,
            inode=metadata.st_ino,
        )
    except OSError as error:
        raise SnapshotError("source read failed") from error
    finally:
        os.close(descriptor)


def _open_snapshot_directory(path: Path) -> int:
    flags = (
        os.O_RDONLY
        | _required_flag("O_DIRECTORY")
        | _required_flag("O_NOFOLLOW")
        | _required_flag("O_CLOEXEC")
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SnapshotError("snapshot directory open failed") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_uid != 0
            or metadata.st_gid != 0
        ):
            raise SnapshotError("snapshot directory is not private")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _write_snapshot(directory_descriptor: int, name: str, contents: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | _required_flag("O_NOFOLLOW")
        | _required_flag("O_CLOEXEC")
    )
    try:
        descriptor = os.open(
            name,
            flags,
            0o600,
            dir_fd=directory_descriptor,
        )
    except OSError as error:
        raise SnapshotError("snapshot creation failed") from error

    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_gid != 0
            or metadata.st_nlink != 1
        ):
            raise SnapshotError("snapshot identity is invalid")
        offset = 0
        while offset < len(contents):
            written = os.write(descriptor, contents[offset:])
            if written <= 0:
                raise SnapshotError("snapshot write failed")
            offset += written
        os.fchown(descriptor, _READINESS_UID, _READINESS_GID)
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != _READINESS_UID
            or metadata.st_gid != _READINESS_GID
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or metadata.st_size != len(contents)
        ):
            raise SnapshotError("snapshot ownership handoff failed")
    except OSError as error:
        raise SnapshotError("snapshot write failed") from error
    finally:
        os.close(descriptor)


def _remove_snapshot(directory_descriptor: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=directory_descriptor)
    except FileNotFoundError:
        pass


def _snapshot_sources(
    attestation_path: Path,
    evidence_path: Path,
    snapshot_directory: Path,
) -> None:
    attestation = _read_source(
        attestation_path,
        expected_mode=0o600,
        expected_owner_uid=_READINESS_UID,
        expected_owner_gid=_READINESS_GID,
        maximum_size=_ATTESTATION_MAX_BYTES,
    )
    evidence = _read_source(
        evidence_path,
        expected_mode=0o400,
        expected_owner_uid=_READINESS_UID,
        expected_owner_gid=_READINESS_GID,
        maximum_size=_EVIDENCE_MAX_BYTES,
    )
    if (attestation.device, attestation.inode) == (evidence.device, evidence.inode):
        raise SnapshotError("sources must be distinct files")

    directory_descriptor = _open_snapshot_directory(snapshot_directory)
    try:
        try:
            _write_snapshot(
                directory_descriptor,
                _ATTESTATION_SNAPSHOT,
                attestation.contents,
            )
            _write_snapshot(
                directory_descriptor,
                _EVIDENCE_SNAPSHOT,
                evidence.contents,
            )
            os.fsync(directory_descriptor)
        except BaseException:
            _remove_snapshot(directory_descriptor, _ATTESTATION_SNAPSHOT)
            _remove_snapshot(directory_descriptor, _EVIDENCE_SNAPSHOT)
            os.fsync(directory_descriptor)
            raise
    finally:
        os.close(directory_descriptor)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create private descriptor-validated readiness snapshots.",
    )
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--snapshot-directory", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if os.geteuid() != 0 or os.getegid() != 0:
        print("Readiness snapshot failed: helper must run as root", file=sys.stderr)
        return 1
    args = _parse_args(argv)
    try:
        _snapshot_sources(
            args.attestation,
            args.evidence,
            args.snapshot_directory,
        )
    except (OSError, SnapshotError):
        print(
            "Readiness snapshot failed: secure descriptor validation failed",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

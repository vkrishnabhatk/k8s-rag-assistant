#!/usr/bin/env python3
"""Release helper: validate, bump versions across all files, commit, and tag."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    result: subprocess.CompletedProcess[str] = subprocess.run(
        cmd, check=True, text=True, capture_output=True
    )
    return result


def _check_preconditions(new_version: str) -> None:
    if not _SEMVER.match(new_version):
        sys.exit(f"error: '{new_version}' is not valid SemVer — use MAJOR.MINOR.PATCH (e.g. 1.0.0)")

    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    if branch != "main":
        sys.exit(f"error: releases must be cut from 'main' (currently on '{branch}')")

    dirty = _run(["git", "status", "--porcelain"]).stdout.strip()
    if dirty:
        sys.exit("error: working tree is dirty — commit or stash changes first")

    existing_tags = _run(["git", "tag"]).stdout.splitlines()
    if f"v{new_version}" in existing_tags:
        sys.exit(f"error: tag v{new_version} already exists")


def _bump_pyproject(new_version: str) -> Path:
    path = ROOT / "pyproject.toml"
    content = path.read_text()
    updated = re.sub(
        r'^(version\s*=\s*")[^"]*(")',
        rf"\g<1>{new_version}\g<2>",
        content,
        count=1,
        flags=re.MULTILINE,
    )
    path.write_text(updated)
    return path


def _bump_init(new_version: str) -> Path:
    path = ROOT / "src" / "rag_assistant" / "__init__.py"
    path.write_text(f'__version__ = "{new_version}"\n')
    return path


def _bump_chart(new_version: str) -> Path:
    path = ROOT / "deploy" / "helm" / "rag-assistant" / "Chart.yaml"
    content = path.read_text()
    content = re.sub(r"^(version:\s*).*", rf"\g<1>{new_version}", content, flags=re.MULTILINE)
    content = re.sub(r"^(appVersion:\s*).*", rf'\g<1>"{new_version}"', content, flags=re.MULTILINE)
    path.write_text(content)
    return path


def release(new_version: str) -> None:
    _check_preconditions(new_version)

    files = [
        _bump_pyproject(new_version),
        _bump_init(new_version),
        _bump_chart(new_version),
    ]

    _run(["git", "add"] + [str(f) for f in files])
    _run(["git", "commit", "-m", f"chore: release v{new_version}"])
    _run(["git", "tag", "-a", f"v{new_version}", "-m", f"Release v{new_version}"])

    print(f"bumped to {new_version} and tagged v{new_version}")
    print()
    print("push to trigger the release build:")
    print(f"  git push && git push origin v{new_version}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} MAJOR.MINOR.PATCH")
    release(sys.argv[1])

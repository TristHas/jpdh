"""What every result pickle records about the code that made it.

``config["libs"]`` pins DiffHydro, DiffRoute and xtensor (and this repository)
by ``git describe --always --dirty --tags`` of the checkout each module was
imported from, or by the installed distribution's version when the module does
not come from a git checkout.  Nothing here is needed to *run* -- a missing git
binary degrades to the version string or ``"unknown"``.
"""
from __future__ import annotations

import datetime as _dt
import subprocess
from importlib import metadata
from pathlib import Path


def git_describe(path) -> str:
    """``git describe --always --dirty --tags`` of the repository containing
    ``path`` (a module file or directory); ``"unknown"`` when it is not one."""
    p = Path(path).resolve()
    d = p if p.is_dir() else p.parent
    try:
        out = subprocess.run(["git", "-C", str(d), "describe", "--always",
                              "--dirty", "--tags"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def _dist_version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "unknown"


def lib_states() -> dict:
    """``{"diffhydro": ..., "diffroute": ..., "xtensor": ..., "jpdh": ...}``."""
    import diffhydro, diffroute, xtensor  # noqa: E401  (already imported by the run)
    out = {}
    for name, mod, dist in (("diffhydro", diffhydro, "diffhydro"),
                            ("diffroute", diffroute, "diffroute"),
                            ("xtensor", xtensor, "xtensor-datatensor")):
        state = git_describe(mod.__file__)
        out[name] = state if state != "unknown" else _dist_version(dist)
    out["jpdh"] = git_describe(Path(__file__).resolve().parents[1])
    return out


def run_provenance() -> dict:
    """Library states and start time for a run's config."""
    return {
        "libs": lib_states(),
        "started_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }

"""Run provenance: what code, what parameters, what environment, how long.

WHY THIS MODULE EXISTS
----------------------
Nothing in the pipeline recorded what a set of results came from. There was no
git SHA, no library versions, no MPI rank count, no seeds, no timings, and --
worst of the set -- no record of the EFFECTIVE parameters.

That last one matters more than it sounds. On the box path,
src/meshing/box_source.py hardcodes vol_frac = 0.08, filter_radius = 0.6 and
opt_tol = 1e-5, silently overriding the 0.15 / 0.006 / 1e-3 sitting in
config.yaml. A reader reconstructing the study from config.yaml would get the
wrong value for essentially every parameter. The manifest written here records
the dicts the solver actually received, so the config file is never the sole
account of what ran.

Artifacts are additionally routed under output/<stage>/<run_id>/ so a new run
cannot overwrite a previous one's results -- which the fixed CWD-relative
artifact paths (optimized_design.xdmf/.jpg) previously guaranteed it would.
"""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from mpi4py import MPI

logger = logging.getLogger(__name__)


def _run_git(args: list[str]) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def git_info() -> dict:
    """Commit, branch and dirty state of the working tree.

    `dirty` is not a formality: the last recorded run of this project was made
    from a tree with deleted and modified tracked files, so the commit SHA alone
    would not have identified the code that produced it.
    """
    sha = _run_git(["rev-parse", "HEAD"])
    status = _run_git(["status", "--porcelain"])
    return {
        "commit": sha,
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status) if status is not None else None,
        "dirty_paths": status.splitlines() if status else [],
    }


def library_versions() -> dict:
    """Versions of every library whose numerics can move a result."""
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for name, module_path in (
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("mpi4py", "mpi4py"),
        ("petsc4py", "petsc4py"),
        ("dolfinx", "dolfinx"),
        ("ufl", "ufl"),
        ("openturns", "openturns"),
        ("pyvista", "pyvista"),
    ):
        try:
            module = __import__(module_path)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = None
    return versions


def make_run_id(comm: MPI.Comm | None = None) -> str:
    """UTC timestamp + short git SHA, identical on every rank.

    Rank 0 builds it and broadcasts, so every rank writes into the same run
    directory rather than each minting its own timestamp.
    """
    comm = comm or MPI.COMM_WORLD
    if comm.rank == 0:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sha = _run_git(["rev-parse", "--short", "HEAD"]) or "nogit"
        run_id = f"{stamp}_{sha}"
    else:
        run_id = None
    return comm.bcast(run_id, root=0)


def to_serializable(obj):
    """Best-effort conversion of pipeline objects to JSON-friendly values.

    Handles the shapes that actually appear in the fem/opt dicts: numpy scalars
    and arrays, dataclasses (KernelParams, MarginalTransformParams), Paths, and
    the UFL forms / dolfinx Functions / lambdas that must be recorded as a
    description rather than a value.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        # Large arrays (meshes, design fields) are summarized, not inlined --
        # the manifest must stay readable.
        if obj.size > 64:
            return {
                "__ndarray__": True,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "min": float(np.min(obj)) if obj.size else None,
                "max": float(np.max(obj)) if obj.size else None,
                "mean": float(np.mean(obj)) if obj.size else None,
            }
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(v) for v in obj]
    if callable(obj):
        return f"<callable {getattr(obj, '__name__', repr(obj))}>"
    return f"<{type(obj).__module__}.{type(obj).__name__}>"


class RunManifest:
    """Accumulates everything needed to reproduce and audit one run.

    Usage:
        manifest = RunManifest(run_id, comm)
        manifest.record_config(cfg, effective_fem=fem, effective_opt=opt)
        with manifest.stage("stage2_fea"):
            ...
        manifest.record("gates", gate_payload)
        manifest.write(path)
    """

    def __init__(self, run_id: str, comm: MPI.Comm | None = None):
        self.comm = comm or MPI.COMM_WORLD
        self.run_id = run_id
        self._start = time.time()
        self._data: dict = {
            "run_id": run_id,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "git": git_info() if self.comm.rank == 0 else None,
            "libraries": library_versions() if self.comm.rank == 0 else None,
            "mpi": {
                "world_size": self.comm.size,
                "vendor": str(MPI.get_vendor()),
            },
            "stage_seconds": {},
        }

    # -- recording -------------------------------------------------------
    def record(self, key: str, value) -> None:
        self._data[key] = to_serializable(value)

    def record_config(self, project_config, effective_fem: dict, effective_opt: dict) -> None:
        """Snapshot BOTH the declared config and the dicts the solver received.

        The two are recorded separately and deliberately: where they disagree,
        the effective values are what produced the results, and the disagreement
        itself is a finding worth seeing in the manifest.
        """
        self._data["config_declared"] = to_serializable(project_config)
        self._data["fem_effective"] = to_serializable(effective_fem)
        self._data["opt_effective"] = to_serializable(effective_opt)

        overrides = _detect_config_overrides(project_config, effective_opt)
        self._data["config_overridden_by_code"] = overrides
        if overrides and self.comm.rank == 0:
            logger.warning(
                "%d config value(s) are OVERRIDDEN in code and did not affect "
                "this run as written in config.yaml: %s. The effective values "
                "are recorded in the manifest; cite those, not the YAML.",
                len(overrides), ", ".join(sorted(overrides)),
            )

    def record_seeds(self, **seeds) -> None:
        self._data["seeds"] = {k: to_serializable(v) for k, v in seeds.items()}

    class _StageTimer:
        def __init__(self, manifest: "RunManifest", name: str):
            self.manifest, self.name = manifest, name

        def __enter__(self):
            self.t0 = time.time()
            return self

        def __exit__(self, *exc):
            self.manifest._data["stage_seconds"][self.name] = time.time() - self.t0
            return False

    def stage(self, name: str) -> "_StageTimer":
        return RunManifest._StageTimer(self, name)

    # -- output ----------------------------------------------------------
    def write(self, path: Path) -> None:
        """Write the manifest (rank 0 only)."""
        if self.comm.rank != 0:
            return
        self._data["finished_utc"] = datetime.now(timezone.utc).isoformat()
        self._data["total_seconds"] = time.time() - self._start
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(self._data, handle, indent=2, default=str)
        logger.info("Run manifest written to %s", path)


# Config fields that the box path deliberately hardcodes, mapped to the opt-dict
# key carrying the value the solver actually used. Listing them explicitly (as
# opposed to diffing everything) keeps the manifest's override report focused on
# the ones that change physics or reported results.
_OVERRIDE_CHECKS = {
    "vol_frac": "vol_frac",
    "filter_radius": "filter_radius",
    "opt_tol": "opt_tol",
    "penalty": "penalty",
    "move": "move",
    "beta_interval": "beta_interval",
    "beta_max": "beta_max",
    "use_oc": "use_oc",
    "opt_compliance": "opt_compliance",
}


def _detect_config_overrides(project_config, effective_opt: dict) -> dict:
    """Report every optimization parameter whose effective value differs from
    the one declared in config.yaml."""
    overrides = {}
    declared = getattr(project_config, "optimization", None)
    if declared is None:
        return overrides
    for config_field, opt_key in _OVERRIDE_CHECKS.items():
        if not hasattr(declared, config_field) or opt_key not in effective_opt:
            continue
        declared_value = getattr(declared, config_field)
        effective_value = effective_opt[opt_key]
        try:
            same = bool(np.isclose(float(declared_value), float(effective_value)))
        except (TypeError, ValueError):
            same = declared_value == effective_value
        if not same:
            overrides[config_field] = {
                "declared_in_config": to_serializable(declared_value),
                "effective_in_run": to_serializable(effective_value),
            }
    return overrides

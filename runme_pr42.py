#!/usr/bin/env python3
"""Lightweight 10 km MISMIP runner for Gadi using local ISSM and pyISSM."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
PYISSM_SRC = Path("/g/data/au88/jr5971/pyISSM/src")
ISSM_PYTHON_DIR = Path(
    "/g/data/vk83/prerelease/apps/spack/1.1/release/linux-x86_64/", 
    "issm-issm-nuopc-v1-2lqdkdualv6jpdkeudkgzlfblzgnhvun", 
    
    # "/g/data/vk83/apps/spack/0.22/release/linux-rocky8-x86_64/gcc-13.2.0/"
    # "issm-git.2025.11.24_2025.11.24-pd55xlx56v5vuno2lshenunfddfvupnr"
)
LOCAL_ISSM_BIN = Path(
    ISSM_PYTHON_DIR / "bin"
    # "/g/data/au88/jr5971/spack/1.1/release/linux-x86_64/"
    # "issm-access-development-hgpsrwrkvxkj4jernyyg3z7tspzpx2zb/bin"
)
MEDIATOR_CASES_DIR = Path("/scratch/au88/jr5971/issm_simple_mediator/cases")

MODEL_NAME = "10km_viscous"
HMAX = 10_000
TIME_STEP_YEARS = 1
FINAL_TIME_YEARS = 20_000
OUTPUT_FREQUENCY_YEARS = 500
CHECKPOINT_FREQUENCY_YEARS = 500
SOLVER_WALLTIME_MINUTES = 60
MPI_RANKS = 8
MEMORY_GB = 64
QUEUE_NAME = "normal"

RUN_NAME = f"MISMIP_{MODEL_NAME}_pr42"
RUN_DIR = BASE_DIR / "rundir"
MESH_MODEL_PATH = RUN_DIR / f"{MODEL_NAME}.issm"
PARAMETERIZED_MODEL_PATH = RUN_DIR / f"{MODEL_NAME}_loaded.issm.nc"
PARAMETER_FILE = BASE_DIR / "Mismip.py"
DOMAIN_FILE = BASE_DIR / "Domain.exp"
FRONT_FILE = BASE_DIR / "Front.exp"

GADI_CONFIG = {
    "login": "jr5971",
    "project": "au88",
    "storage": "gdata/au88+gdata/vk83+gdata/xp65+scratch/au88",
    "executionpath": "/scratch/au88/jr5971/issm_runs",
    "codepath": str(LOCAL_ISSM_BIN),
    "np": MPI_RANKS,
    "memory_gb": MEMORY_GB,
    "queue": QUEUE_NAME,
    "time_minutes": SOLVER_WALLTIME_MINUTES,
    "moduleuse": [
        "/g/data/vk83/prerelease/modules",
        "/g/data/xp65/public/modules",
        # "/apps/Modules/modulefiles",
    ],
    "moduleload": [
        "access-issm/pr42-1",
        "conda/analysis3-26.02",
        # "gcc/13.2.0",
        # "openmpi/4.1.5",
    ],
}
PBS_BIN = "/opt/pbs/default/bin"


def bootstrap_paths() -> None:
    """Make pyISSM and the wrapper-enabled ISSM Python runtime importable."""

    os.environ["ISSM_DIR"] = str(ISSM_PYTHON_DIR)
    path_entries = os.environ.get("PATH", "").split(":") if os.environ.get("PATH") else []
    if PBS_BIN not in path_entries:
        os.environ["PATH"] = f"{PBS_BIN}:{os.environ.get('PATH', '')}".rstrip(":")

    for path in (
        PYISSM_SRC,
        ISSM_PYTHON_DIR / "python-tools.zip",
        ISSM_PYTHON_DIR / "lib",
    ):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


bootstrap_paths()

import ctypes
import pyissm  # noqa: E402

ctypes.CDLL(
    str(ISSM_PYTHON_DIR / "lib" / "libISSMNUOPCCap.so"), 
    mode = ctypes.RTLD_GLOBAL, 
)



def build_cluster():
    """Create a Gadi cluster object that targets the local ISSM executable."""

    cluster = pyissm.model.classes.cluster.gadi()
    cluster.name = pyissm.tools.config.get_hostname()
    cluster.login = GADI_CONFIG["login"]
    cluster.project = GADI_CONFIG["project"]
    cluster.storage = GADI_CONFIG["storage"]
    cluster.executionpath = GADI_CONFIG["executionpath"]
    cluster.codepath = GADI_CONFIG["codepath"]
    cluster.np = GADI_CONFIG["np"]
    cluster.memory = GADI_CONFIG["memory_gb"]
    cluster.queue = GADI_CONFIG["queue"]
    cluster.time = GADI_CONFIG["time_minutes"]
    cluster.moduleuse = list(GADI_CONFIG["moduleuse"])
    cluster.moduleload = list(GADI_CONFIG["moduleload"])
    return cluster


def prepare_mesh():
    """Build and save the lightweight 10 km mesh."""

    md = pyissm.model.Model()
    md = pyissm.model.mesh.bamg(
        md,
        domain=str(DOMAIN_FILE),
        hmax=HMAX,
        splitcorners=1,
    )
    md.miscellaneous.name = f"MISMIP_{MODEL_NAME}"
    pyissm.model.io.save_model(md, str(MESH_MODEL_PATH))
    return md


def parameterize_model():
    """Reload the mesh model, apply MISMIP physics, and persist the configured state."""

    md = pyissm.model.io.load_model(str(MESH_MODEL_PATH))
    md = pyissm.model.param.parameterize(md, str(PARAMETER_FILE))
    return md


def configure_transient(md):
    """Configure the lightweight SSA transient spin-up."""

    md = pyissm.model.param.set_flow_equation(md, SSA="all")
    md.timestepping.time_step = TIME_STEP_YEARS
    md.timestepping.final_time = FINAL_TIME_YEARS
    md.settings.output_frequency = OUTPUT_FREQUENCY_YEARS
    md.settings.checkpoint_frequency = CHECKPOINT_FREQUENCY_YEARS
    md.stressbalance.maxiter = 30
    md.settings.solver_residue_threshold = np.nan
    md.cluster = build_cluster()
    md.miscellaneous.name = RUN_NAME
    return md


def save_presolve_model(md) -> None:
    """Write the configured pre-solve model for inspection and reuse."""

    pyissm.model.io.save_model(md, str(PARAMETERIZED_MODEL_PATH))


def validate_inputs() -> None:
    """Ensure the source assets needed to build the lightweight case are present."""

    if not DOMAIN_FILE.exists():
        raise FileNotFoundError(f"Domain contour not found: {DOMAIN_FILE}")
    if not FRONT_FILE.exists():
        raise FileNotFoundError(f"Ice front contour not found: {FRONT_FILE}")
    if not PARAMETER_FILE.exists():
        raise FileNotFoundError(f"Parameterisation file not found: {PARAMETER_FILE}")


def build_configured_model():
    """Prepare the lightweight mesh, parameterisation, and transient settings."""

    os.chdir(BASE_DIR)
    RUN_DIR.mkdir(exist_ok=True)
    validate_inputs()

    print("Preparing lightweight MISMIP mesh...")
    prepare_mesh()

    print("Parameterising MISMIP configuration...")
    md = parameterize_model()

    print("Configuring SSA transient run...")
    md = configure_transient(md)
    save_presolve_model(md)
    return md


def remove_existing_case_artifacts(case_dir: Path, model_name: str) -> None:
    """Remove stale files for the staged mediator case without touching other cases."""

    prefixes = (f"{model_name}.", f"{model_name}_")
    for path in case_dir.iterdir():
        if not path.name.startswith(prefixes):
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def stage_mediator_case(case_root: Path = MEDIATOR_CASES_DIR) -> Path:
    """Marshall a clean `.bin`/`.toolkits` case for the external mediator."""

    md = build_configured_model()
    model_name = md.miscellaneous.name
    case_dir = case_root / model_name
    case_dir.mkdir(parents=True, exist_ok=True)
    remove_existing_case_artifacts(case_dir, model_name)

    md.private.solution = "TransientSolution"
    pyissm.model.execute.is_model_self_consistent(md)

    cwd = Path.cwd()
    try:
        os.chdir(case_dir)
        pyissm.model.execute.marshall(md)
        md.toolkits.write_toolkits_file(f"{model_name}.toolkits")
    finally:
        os.chdir(cwd)

    print(f"Staged mediator case: {case_dir}")
    return case_dir


def submit_standalone_run():
    """Prepare, configure, and submit the lightweight standalone transient run."""

    md = build_configured_model()

    print(f"Submitting {RUN_NAME} with {md.cluster.np} MPI ranks...")
    md = pyissm.model.execute.solve(
        md=md,
        solution_string="tr",
        load_only=False,
        runtime_name=False,
    )

    return md


def parse_args() -> argparse.Namespace:
    """Parse a small command surface for standalone or mediator workflows."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "submit",
        help="Submit the lightweight standalone ISSM transient run.",
    )

    stage_parser = subparsers.add_parser(
        "stage-mediator-case",
        help="Write a clean mediator case directory with only .bin and .toolkits.",
    )
    stage_parser.add_argument(
        "--case-root",
        type=Path,
        default=MEDIATOR_CASES_DIR,
        help="Directory under which the mediator case directory will be created.",
    )

    return parser.parse_args()


def main():
    """Dispatch the lightweight workflow for standalone or mediator usage."""

    args = parse_args()
    if args.command in (None, "submit"):
        submit_standalone_run()
        return

    if args.command == "stage-mediator-case":
        stage_mediator_case(args.case_root)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

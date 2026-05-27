#!/usr/bin/env python3
"""Prepare and optionally run a small NUOPC-driven MISMIP experiment suite."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import runme_pr42


BASE_DIR = Path(__file__).resolve().parent
MEDIATOR_ROOT = Path("/g/data/au88/jr5971/issm_simple_mediator")
DEFAULT_MEDIATOR_BUILD_DIR = "build-local-issm"
EXPERIMENT_ROOT = Path("/scratch/au88/jr5971/issm_simple_mediator/experiments")
GENERATED_CONFIG_DIR = MEDIATOR_ROOT / "config" / "generated"
FIELD_DICTIONARY = MEDIATOR_ROOT / "config" / "issm_field_dictionary.yaml"
CATALOG_PATH = EXPERIMENT_ROOT / "catalog.csv"
LOCAL_ESMFMKFILE = (
    "/g/data/au88/jr5971/spack/1.1/release/linux-x86_64/"
    "esmf-8.7.0-3qpuyn32m6ff6ikhov6jpsqgpmafq23a/lib/esmf.mk"
)
PR42_ESMFMKFILE = (
    "/g/data/vk83/prerelease/apps/spack/1.1/release/linux-x86_64/"
    "esmf-8.7.0-i5eigjwgwpaec35udpsbcf3a5i6wjuox/lib/esmf.mk"
)

SECONDS_PER_YEAR = 31_557_600
DEFAULT_MPI_RANKS = 8


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    melt_rate_m_per_s: float
    coupling_years: int
    purpose: str
    suite: str
    final_time_years: int
    output_frequency_years: int = 5
    checkpoint_frequency_years: int = 5
    time_step_years: int = 1

    @property
    def model_name(self) -> str:
        return f"MISMIP_10km_viscous_{self.experiment_id}"

    @property
    def experiment_dir(self) -> Path:
        return EXPERIMENT_ROOT / self.experiment_id

    @property
    def case_dir(self) -> Path:
        return self.experiment_dir / "case"

    @property
    def config_path(self) -> Path:
        return GENERATED_CONFIG_DIR / f"issm_simple_mediator_{self.experiment_id}.yaml"

    @property
    def pre_solve_nc(self) -> Path:
        return self.experiment_dir / "pre_solve.nc"

    @property
    def outbin_path(self) -> Path:
        return self.case_dir / f"{self.model_name}.outbin"

    @property
    def result_nc(self) -> Path:
        return self.experiment_dir / "result.nc"

    @property
    def time_step_seconds(self) -> int:
        return self.coupling_years * SECONDS_PER_YEAR

    @property
    def run_steps(self) -> int:
        if self.final_time_years % self.coupling_years:
            raise ValueError(
                f"{self.experiment_id}: final_time_years must be divisible by coupling_years"
            )
        return self.final_time_years // self.coupling_years

    @property
    def melt_rate_m_per_yr(self) -> float:
        return self.melt_rate_m_per_s * SECONDS_PER_YEAR


def full_suite() -> list[Experiment]:
    return [
        Experiment("melt_0_cpl10y", 0.0, 10, "control", "full", 200),
        Experiment("melt_0p1myr_cpl10y", 3.16887646e-9, 10, "low melt sensitivity", "full", 200),
        Experiment("melt_1myr_cpl10y", 3.16887646e-8, 10, "moderate melt sensitivity", "full", 200),
        Experiment("melt_5myr_cpl10y", 1.58443823e-7, 10, "high melt sensitivity", "full", 200),
        Experiment("melt_1myr_cpl5y", 3.16887646e-8, 5, "temporal coupling comparison", "full", 200),
    ]


def dry_suite() -> list[Experiment]:
    return [
        Experiment("melt_0_cpl10y", 0.0, 10, "20-year dry control", "dry", 20),
        Experiment("melt_1myr_cpl10y", 3.16887646e-8, 10, "20-year dry melt test", "dry", 20),
    ]


def selected_suite(name: str) -> list[Experiment]:
    if name == "dry":
        return dry_suite()
    if name == "full":
        return full_suite()
    raise ValueError(f"Unsupported suite: {name}")


def write_mediator_config(experiment: Experiment) -> None:
    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_text = f"""Mediator:

  App:
    fieldDictionary: {FIELD_DICTIONARY}
    logFlush: true
    startTime: 2000-01-01T00:00:00
    timeStepSeconds: {experiment.time_step_seconds}
    runSteps: {experiment.run_steps}

  Driver:
    runSequence: |
      @{experiment.time_step_seconds}
        Forcing
        Forcing -> ISSM
        ISSM
      @

  ISSM:
    attributes:
      case_dir: {experiment.case_dir}
      model_name: {experiment.model_name}
      solution_name: TransientSolution
      write_restart: "false"

  Forcing:
    mode: constant
    constantBasalMeltRate: {experiment.melt_rate_m_per_s:.10g}

  Output:
    summaryEveryStep: true
"""
    experiment.config_path.write_text(config_text, encoding="utf-8")
    shutil.copy2(experiment.config_path, experiment.experiment_dir / "mediator_config.yaml")


def configure_model(experiment: Experiment):
    md = runme_pr42.parameterize_model()
    md = runme_pr42.pyissm.model.param.set_flow_equation(md, SSA="all")
    md.timestepping.time_step = experiment.time_step_years
    md.timestepping.final_time = experiment.final_time_years
    md.settings.output_frequency = experiment.output_frequency_years
    md.settings.checkpoint_frequency = experiment.checkpoint_frequency_years
    md.stressbalance.maxiter = 30
    md.settings.solver_residue_threshold = np.nan
    md.cluster = runme_pr42.build_cluster()
    md.miscellaneous.name = experiment.model_name
    md.private.solution = "TransientSolution"
    return md


def stage_case(experiment: Experiment) -> None:
    os.chdir(BASE_DIR)
    runme_pr42.RUN_DIR.mkdir(exist_ok=True)
    runme_pr42.validate_inputs()
    experiment.experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment.case_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in (
        experiment.result_nc,
        experiment.experiment_dir / "mediator.log",
        experiment.experiment_dir / "mediator_stdout.txt",
        experiment.experiment_dir / "mediator_stderr.txt",
    ):
        if stale_path.exists():
            stale_path.unlink()

    if not runme_pr42.MESH_MODEL_PATH.exists():
        print("Preparing shared 10 km MISMIP mesh...")
        runme_pr42.prepare_mesh()

    print(f"Parameterising and staging {experiment.experiment_id}...")
    md = configure_model(experiment)
    runme_pr42.pyissm.model.io.save_model(md, str(experiment.pre_solve_nc))
    runme_pr42.remove_existing_case_artifacts(experiment.case_dir, experiment.model_name)
    runme_pr42.pyissm.model.execute.is_model_self_consistent(md)

    cwd = Path.cwd()
    try:
        os.chdir(experiment.case_dir)
        runme_pr42.pyissm.model.execute.marshall(md)
        md.toolkits.write_toolkits_file(f"{experiment.model_name}.toolkits")
    finally:
        os.chdir(cwd)

    write_mediator_config(experiment)


def run_command_text(experiment: Experiment, mediator_build_dir: str) -> str:
    if mediator_build_dir == "build-pr42":
        module_setup = f"""
module use /g/data/vk83/prerelease/modules
module load access-issm/pr42-1
export ESMFMKFILE="{PR42_ESMFMKFILE}"
"""
    else:
        module_setup = f"""
module load gcc/13.2.0 openmpi/4.1.5
export ISSM_DIR="${{ISSM_DIR:-/g/data/au88/jr5971/ISSM}}"
export ESMFMKFILE="{LOCAL_ESMFMKFILE}"
"""
    return f"""
set -euo pipefail
module purge
{module_setup}
export MEDIATOR_BUILD_DIR="{mediator_build_dir}"
export MPI_RANKS="${{MPI_RANKS:-{DEFAULT_MPI_RANKS}}}"
./run_mediator.sh "{experiment.config_path}"
"""


def parse_mediator_log_dir(stdout: str) -> str:
    for line in stdout.splitlines():
        prefix = "Mediator run directory: "
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def run_mediator(experiment: Experiment, timeout_seconds: int, mediator_build_dir: str) -> dict[str, object]:
    command = run_command_text(experiment, mediator_build_dir)
    print(f"Running mediator for {experiment.experiment_id}...")
    process = subprocess.Popen(
        ["bash", "-lc", command],
        cwd=MEDIATOR_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    timed_out = False
    try:
        stdout, stderr = process.communicate(
            timeout=timeout_seconds if timeout_seconds > 0 else None
        )
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()

    (experiment.experiment_dir / "mediator_stdout.txt").write_text(stdout, encoding="utf-8")
    (experiment.experiment_dir / "mediator_stderr.txt").write_text(stderr, encoding="utf-8")
    log_dir = parse_mediator_log_dir(stdout)
    log_path = Path(log_dir) / "mediator.log" if log_dir else None
    if log_path and log_path.exists():
        shutil.copy2(log_path, experiment.experiment_dir / "mediator.log")

    return {
        "returncode": process.returncode,
        "timed_out": timed_out,
        "mediator_log_dir": log_dir,
        "mediator_log": str(log_path) if log_path else "",
        "mediator_build_dir": mediator_build_dir,
        "mediator_executable": str(MEDIATOR_ROOT / mediator_build_dir / "issm-simple-mediator"),
    }


def convert_outbin(experiment: Experiment) -> tuple[str, str]:
    if not experiment.outbin_path.exists():
        return "missing_outbin", ""

    print(f"Converting {experiment.experiment_id} outbin to NetCDF...")
    md = runme_pr42.pyissm.model.io.load_model(str(experiment.pre_solve_nc))
    cwd = Path.cwd()
    try:
        os.chdir(experiment.case_dir)
        md = runme_pr42.pyissm.model.execute.load_results_from_disk(
            md, str(experiment.outbin_path)
        )
        runme_pr42.pyissm.model.io.save_model(md, str(experiment.result_nc))
    except Exception as exc:  # noqa: BLE001
        return "conversion_failed", str(exc)
    finally:
        os.chdir(cwd)
    return "converted", ""


def write_metadata(experiment: Experiment, run_info: dict[str, object], conversion_status: str, error: str) -> dict[str, object]:
    metadata = {
        **asdict(experiment),
        "model_name": experiment.model_name,
        "melt_rate_m_per_yr": experiment.melt_rate_m_per_yr,
        "time_step_seconds": experiment.time_step_seconds,
        "run_steps": experiment.run_steps,
        "experiment_dir": str(experiment.experiment_dir),
        "case_dir": str(experiment.case_dir),
        "config_path": str(experiment.config_path),
        "pre_solve_nc": str(experiment.pre_solve_nc),
        "outbin_path": str(experiment.outbin_path),
        "result_nc": str(experiment.result_nc),
        "conversion_status": conversion_status,
        "conversion_error": error,
        **run_info,
    }

    if conversion_status == "not_run":
        metadata["status"] = "prepared"
    elif run_info.get("timed_out"):
        metadata["status"] = "timeout"
    elif run_info.get("returncode", 0) not in (0, None):
        metadata["status"] = "failed"
    elif conversion_status == "converted":
        metadata["status"] = "complete"
    else:
        metadata["status"] = conversion_status

    (experiment.experiment_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def write_catalog(rows: list[dict[str, object]]) -> None:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = [
        "experiment_id",
        "suite",
        "status",
        "purpose",
        "melt_rate_m_per_s",
        "melt_rate_m_per_yr",
        "coupling_years",
        "time_step_seconds",
        "run_steps",
        "final_time_years",
        "output_frequency_years",
        "checkpoint_frequency_years",
        "time_step_years",
        "returncode",
        "timed_out",
        "conversion_status",
        "conversion_error",
        "mediator_build_dir",
        "mediator_executable",
        "experiment_dir",
        "case_dir",
        "config_path",
        "mediator_log",
        "mediator_log_dir",
        "outbin_path",
        "result_nc",
        "pre_solve_nc",
    ]
    lock_path = CATALOG_PATH.with_name(f"{CATALOG_PATH.name}.lock")
    tmp_path = CATALOG_PATH.with_name(f".{CATALOG_PATH.name}.{os.getpid()}.tmp")

    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        merged_rows: dict[str, dict[str, object]] = {}
        if CATALOG_PATH.exists():
            with CATALOG_PATH.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    experiment_id = row.get("experiment_id", "")
                    if experiment_id:
                        merged_rows[experiment_id] = dict(row)
        for row in rows:
            experiment_id = str(row.get("experiment_id", ""))
            if experiment_id:
                merged_rows[experiment_id] = row

        with tmp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in merged_rows.values():
                writer.writerow({key: row.get(key, "") for key in fieldnames})
        os.replace(tmp_path, CATALOG_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("dry", "full"), default="dry")
    parser.add_argument(
        "--experiment",
        action="append",
        help="Run only a named experiment. May be passed more than once.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the mediator after preparing cases and configs.",
    )
    parser.add_argument(
        "--mediator-build-dir",
        default=DEFAULT_MEDIATOR_BUILD_DIR,
        help="Mediator build directory under /g/data/au88/jr5971/issm_simple_mediator.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=0,
        help="Per-experiment mediator timeout. Use 0 for no timeout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiments = selected_suite(args.suite)
    if args.experiment:
        wanted = set(args.experiment)
        experiments = [experiment for experiment in experiments if experiment.experiment_id in wanted]
        missing = wanted.difference({experiment.experiment_id for experiment in experiments})
        if missing:
            raise ValueError(f"Unknown experiments for suite {args.suite}: {sorted(missing)}")

    mediator_build_path = MEDIATOR_ROOT / args.mediator_build_dir
    if not mediator_build_path.exists():
        raise FileNotFoundError(f"Expected mediator build: {mediator_build_path}")

    rows: list[dict[str, object]] = []
    for experiment in experiments:
        stage_case(experiment)
        run_info: dict[str, object] = {
            "returncode": None,
            "timed_out": False,
            "mediator_log_dir": "",
            "mediator_log": "",
            "mediator_build_dir": args.mediator_build_dir,
            "mediator_executable": str(mediator_build_path / "issm-simple-mediator"),
        }
        conversion_status = "not_run"
        conversion_error = ""

        if args.execute:
            run_info = run_mediator(experiment, args.timeout_seconds, args.mediator_build_dir)
            if run_info.get("returncode") == 0 and not run_info.get("timed_out"):
                conversion_status, conversion_error = convert_outbin(experiment)
            else:
                conversion_status = "skipped_failed_run"

        rows.append(write_metadata(experiment, run_info, conversion_status, conversion_error))
        write_catalog(rows)

    print(f"Wrote catalog: {CATALOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS_SCRIPT="${SCRIPT_DIR}/run_mediator_experiment_suite.pbs"
MISMIP_SUITE_ROOT="${SCRIPT_DIR}"

PROJECT="au88"
QUEUE="normal"
WALLTIME="06:00:00"
NCPUS="8"
MEM="32GB"
STORAGE="gdata/au88+gdata/vk83+gdata/xp65+scratch/au88"
JOB_NAME="issm_mediator_suite"
PBS_LOG_ROOT="/scratch/au88/jr5971/issm_simple_mediator/pbs"

SUITE="full"
EXECUTE="1"
TIMEOUT_SECONDS="7200"
MEDIATOR_BUILD_DIR="build-local-issm"
MPI_RANKS=""
OMP_NUM_THREADS="1"
REBUILD_LOCAL_ISSM="0"
DRY_RUN="0"
PARALLEL="0"
EXPERIMENTS=()

usage() {
  cat <<'EOF'
Submit the NUOPC-driven MISMIP mediator experiment suite to PBS.

Examples:
  # Submit the full 200-year suite using the patched local mediator build.
  ./submit_mediator_experiment_suite.sh

  # Submit each full-suite experiment as a separate PBS job.
  ./submit_mediator_experiment_suite.sh --parallel

  # Submit only the full-suite control.
  ./submit_mediator_experiment_suite.sh --experiment melt_0_cpl10y

  # Submit the 20-year dry validation subset.
  ./submit_mediator_experiment_suite.sh --suite dry --walltime 01:00:00

  # Prepare cases/configs on a compute node without running the mediator.
  ./submit_mediator_experiment_suite.sh --suite full --prepare-only

  # Larger resource request for later experiments.
  ./submit_mediator_experiment_suite.sh --ncpus 16 --mpi-ranks 16 --mem 64GB --walltime 12:00:00

Options:
  --suite dry|full              Experiment suite to run. Default: full.
  --experiment ID               Select one experiment. May be repeated.
  --prepare-only                Stage cases/configs without mediator execution.
  --execute                     Execute mediator runs. Default.
  --mediator-build-dir DIR      Mediator build dir under issm_simple_mediator.
                                Default: build-local-issm.
  --timeout-seconds SECONDS     Per-experiment timeout. Use 0 for no timeout.
                                Default: 7200.
  --rebuild                     Rebuild local ISSM cap and mediator inside the job.
  --project PROJECT             PBS project. Default: au88.
  --queue QUEUE                 PBS queue. Default: normal.
  --walltime HH:MM:SS           PBS walltime. Default: 06:00:00.
  --ncpus N                     PBS ncpus. Default: 8.
  --mpi-ranks N                 MPI ranks used by run_mediator.sh. Default: ncpus.
  --mem SIZE                    PBS memory. Default: 32GB.
  --storage LIST                PBS storage request.
  --job-name NAME               PBS job name. Default: issm_mediator_suite.
  --pbs-log-root PATH           PBS stdout/qstat output directory.
  --parallel                    Submit one PBS job per selected experiment.
  --dry-run                     Print qsub command without submitting.
  -h, --help                    Show this help.
EOF
}

default_experiments_for_suite() {
  case "$1" in
    dry)
      printf '%s\n' melt_0_cpl10y melt_1myr_cpl10y
      ;;
    full)
      printf '%s\n' \
        melt_0_cpl10y \
        melt_0p1myr_cpl10y \
        melt_1myr_cpl10y \
        melt_5myr_cpl10y \
        melt_1myr_cpl5y
      ;;
    *)
      return 1
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)
      SUITE="$2"
      shift 2
      ;;
    --experiment)
      EXPERIMENTS+=("$2")
      shift 2
      ;;
    --prepare-only)
      EXECUTE="0"
      shift
      ;;
    --execute)
      EXECUTE="1"
      shift
      ;;
    --mediator-build-dir)
      MEDIATOR_BUILD_DIR="$2"
      shift 2
      ;;
    --timeout-seconds)
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --rebuild)
      REBUILD_LOCAL_ISSM="1"
      shift
      ;;
    --project)
      PROJECT="$2"
      shift 2
      ;;
    --queue)
      QUEUE="$2"
      shift 2
      ;;
    --walltime)
      WALLTIME="$2"
      shift 2
      ;;
    --ncpus)
      NCPUS="$2"
      shift 2
      ;;
    --mpi-ranks)
      MPI_RANKS="$2"
      shift 2
      ;;
    --mem)
      MEM="$2"
      shift 2
      ;;
    --storage)
      STORAGE="$2"
      shift 2
      ;;
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    --pbs-log-root)
      PBS_LOG_ROOT="$2"
      shift 2
      ;;
    --parallel)
      PARALLEL="1"
      shift
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${SUITE}" in
  dry|full)
    ;;
  *)
    echo "ERROR: --suite must be 'dry' or 'full', got '${SUITE}'." >&2
    exit 2
    ;;
esac

if [[ ! -f "${PBS_SCRIPT}" ]]; then
  echo "ERROR: PBS script not found: ${PBS_SCRIPT}" >&2
  exit 2
fi

if [[ -z "${MPI_RANKS}" ]]; then
  MPI_RANKS="${NCPUS}"
fi

if [[ "${PARALLEL}" == "1" && "${REBUILD_LOCAL_ISSM}" == "1" ]]; then
  echo "ERROR: --parallel and --rebuild cannot be used together." >&2
  echo "Rebuild once first, then submit parallel experiment jobs." >&2
  exit 2
fi

mkdir -p "${PBS_LOG_ROOT}"

if ! command -v qsub >/dev/null 2>&1; then
  export PATH="/opt/pbs/default/bin:${PATH}"
fi

submit_one_job() {
  local experiments_joined="$1"
  local job_name="$2"
  local var_list
  var_list=$(
    printf 'MISMIP_SUITE_ROOT=%s,SUITE=%s,EXPERIMENTS=%s,EXECUTE=%s,MEDIATOR_BUILD_DIR=%s,TIMEOUT_SECONDS=%s,MPI_RANKS=%s,OMP_NUM_THREADS=%s,REBUILD_LOCAL_ISSM=%s,PBS_LOG_ROOT=%s' \
      "${MISMIP_SUITE_ROOT}" \
      "${SUITE}" \
      "${experiments_joined}" \
      "${EXECUTE}" \
      "${MEDIATOR_BUILD_DIR}" \
      "${TIMEOUT_SECONDS}" \
      "${MPI_RANKS}" \
      "${OMP_NUM_THREADS}" \
      "${REBUILD_LOCAL_ISSM}" \
      "${PBS_LOG_ROOT}"
  )

  local qsub_cmd=(
    qsub
    -P "${PROJECT}"
    -q "${QUEUE}"
    -N "${job_name}"
    -l "walltime=${WALLTIME},ncpus=${NCPUS},mem=${MEM},storage=${STORAGE}"
    -j oe
    -o "${PBS_LOG_ROOT}"
    -v "${var_list}"
    "${PBS_SCRIPT}"
  )

  printf 'qsub command:'
  printf ' %q' "${qsub_cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" != "1" ]]; then
    "${qsub_cmd[@]}"
  fi
}

if [[ "${PARALLEL}" == "1" ]]; then
  if ((${#EXPERIMENTS[@]} == 0)); then
    mapfile -t EXPERIMENTS < <(default_experiments_for_suite "${SUITE}")
  fi

  for experiment_id in "${EXPERIMENTS[@]}"; do
    submit_one_job "${experiment_id}" "${JOB_NAME}"
  done
else
  experiments_joined=""
  if ((${#EXPERIMENTS[@]})); then
    old_ifs="${IFS}"
    IFS=':'
    experiments_joined="${EXPERIMENTS[*]}"
    IFS="${old_ifs}"
  fi
  submit_one_job "${experiments_joined}" "${JOB_NAME}"
fi

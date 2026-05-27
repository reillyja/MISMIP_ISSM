# MISMIP_ISSM
MISMIP_ISSM_config

Square floating ice with ice-ocean boundary

## NUOPC Mediator Experiment Suite

The script `run_mediator_experiment_suite.py` prepares and optionally runs a
catalog-driven MISMIP experiment suite through the local `issm_simple_mediator`
driver. It only writes under `/g/data/au88/jr5971` and
`/scratch/au88/jr5971`; `/g/data/vk83` is a read-only dependency location for
this workflow.

Build the local ISSM-linked mediator after changing the local NUOPC cap or
mediator source:

```bash
/g/data/au88/jr5971/issm_simple_mediator/build_local_issm.sh
```

### Running On Compute Nodes

Use the PBS wrapper for full experiments. It submits
`run_mediator_experiment_suite.py` to a compute node and defaults to the fixed
`build-local-issm` mediator.

Submit the full 200-year suite:

```bash
cd /g/data/au88/jr5971/MISMIP_ISSM
./submit_mediator_experiment_suite.sh
```

Submit the full 200-year suite as separate PBS jobs, one per experiment:

```bash
./submit_mediator_experiment_suite.sh --parallel
```

Submit one full-suite experiment first:

```bash
./submit_mediator_experiment_suite.sh --experiment melt_0_cpl10y
```

Submit the 20-year dry validation subset:

```bash
./submit_mediator_experiment_suite.sh --suite dry --walltime 01:00:00
```

Request larger resources for later experiments:

```bash
./submit_mediator_experiment_suite.sh \
  --ncpus 16 \
  --mpi-ranks 16 \
  --mem 64GB \
  --walltime 12:00:00
```

Preview the `qsub` command without submitting:

```bash
./submit_mediator_experiment_suite.sh --suite full --dry-run
```

Preview all per-experiment `qsub` commands for a parallel suite:

```bash
./submit_mediator_experiment_suite.sh --suite full --parallel --dry-run
```

The wrapper writes PBS output and `qstat` snapshots under
`/scratch/au88/jr5971/issm_simple_mediator/pbs/`. Per-experiment mediator logs
and NetCDF outputs still go under
`/scratch/au88/jr5971/issm_simple_mediator/experiments/` and
`/scratch/au88/jr5971/issm_simple_mediator/logs/`.

### Direct Local Runs

The commands below run in the shell where they are launched. Use them inside an
interactive PBS job or for short preparation steps; do not launch full 200-year
runs from a login node.

Prepare the 20-year dry subset without launching the mediator:

```bash
python /g/data/au88/jr5971/MISMIP_ISSM/run_mediator_experiment_suite.py --suite dry
```

Run the 20-year dry subset:

```bash
python /g/data/au88/jr5971/MISMIP_ISSM/run_mediator_experiment_suite.py \
  --suite dry \
  --execute \
  --timeout-seconds 900
```

Prepare the full 200-year suite:

```bash
python /g/data/au88/jr5971/MISMIP_ISSM/run_mediator_experiment_suite.py --suite full
```

Run the full 200-year suite:

```bash
python /g/data/au88/jr5971/MISMIP_ISSM/run_mediator_experiment_suite.py \
  --suite full \
  --execute \
  --timeout-seconds 7200 \
  --mediator-build-dir build-local-issm
```

The 10-year mediator timing issue exposed on 2026-05-26 was fixed locally by:

- passing the mediator YAML `timeStepSeconds` into the ISSM cap as an explicit
  `advance_seconds` attribute;
- using the same value in generated NUOPC run sequences, for example
  `@315576000` rather than `@1`;
- suppressing ISSM checkpoint/restart reloads inside each in-process NUOPC
  advance while preserving the staged checkpoint-frequency metadata.

The fixed 20-year dry control and 1 m/yr melt cases completed and converted to
NetCDF with `build-local-issm`. The mediator clock uses 365.25-day years, while
the MISMIP case uses `md.constants.yts = 31556926`, so NetCDF output times can
show small offsets such as `20.0004` years after a nominal 20-year mediator run.

The suite writes one experiment directory per case under
`/scratch/au88/jr5971/issm_simple_mediator/experiments/`, generated mediator
YAML files under `/g/data/au88/jr5971/issm_simple_mediator/config/generated/`,
and a catalog at
`/scratch/au88/jr5971/issm_simple_mediator/experiments/catalog.csv`.

Open `mismip_mediator_experiment_suite_visualisation.ipynb` after at least one
experiment has completed. The notebook reads the catalog, loads successful
`result.nc` files, and compares final states, final-minus-control fields,
domain-mean time series, centerline profiles, coupling cadence, and mediator
log provenance.

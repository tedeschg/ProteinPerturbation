#!/bin/bash
###############################################################################
# runit_pipeline.sh — Metacentrum PBS wrapper for run_pipeline.sh
#
# Usage (on a Metacentrum frontend, e.g. skirit/zuphux):
#   bash runit_pipeline.sh <pdb1> [pdb2 ...]
#
# Submits one PBS job that runs the full LigandMPNN pipeline (QC + residue
# selection + design) on the listed PDB files using one GPU.
#
# Outputs (stdout/stderr of the job) are written next to the submission dir.
# Pipeline results land in the output_dir defined in scripts/config.yaml.
###############################################################################

set -euo pipefail

# =============================================================================
# User-configurable paths — EDIT THESE for Metacentrum
# =============================================================================
PROJECT_ROOT="/storage/brno12-cerit/home/tedeschg/prj/protein-perturbation"

# PBS resources
QUEUE="gpu"
WALLTIME="03:00:00"    # ~1h46 on a GTX 1650 → ~20-40 min on mid-tier Metacentrum GPU; 3h is safe.
NCPUS=4
NGPUS=1
GPU_MEM="10gb"         # filter out very old/small GPUs (LigandMPNN works fine on 8GB+)
MEM="32gb"
SCRATCH="20gb"
EXCLUDE_CL="cl_bee=False"   # exclude the bee cluster (older / less stable). Leave empty to allow it.

# =============================================================================
# Validate args
# =============================================================================
if [ $# -eq 0 ]; then
    echo "Usage: $0 <pdb1.pdb> [pdb2.pdb ...]" >&2
    exit 1
fi

# Resolve input PDBs to absolute paths (so they're valid inside the job)
INPUT_PDBS=()
for f in "$@"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: file not found: $f" >&2
        exit 1
    fi
    INPUT_PDBS+=("$(realpath "$f")")
done

# Build space-separated, quoted PDB arguments string for the inner script
PDB_ARGS=""
for p in "${INPUT_PDBS[@]}"; do
    PDB_ARGS+=" \"${p}\""
done

MYDIR="$(pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_FILE="${MYDIR}/pipeline_${TIMESTAMP}.$$.pbs"

# =============================================================================
# Generate PBS script (heredoc; \$VAR is escaped to be expanded by the job,
# ${VAR} is expanded NOW by this wrapper)
# =============================================================================
cat > "$RUN_FILE" <<EOF
#!/bin/sh
#PBS -q ${QUEUE}
#PBS -N pipeline_${TIMESTAMP}
#PBS -l walltime=${WALLTIME}
#PBS -l select=1:ngpus=${NGPUS}:ncpus=${NCPUS}:mem=${MEM}:scratch_local=${SCRATCH}:gpu_mem=${GPU_MEM}${EXCLUDE_CL:+:${EXCLUDE_CL}}
#PBS -o ${MYDIR}/pipeline_${TIMESTAMP}.stdout
#PBS -e ${MYDIR}/pipeline_${TIMESTAMP}.stderr

set -euo pipefail

# Auto-clean scratch on exit
trap 'rm -rf "\$SCRATCHDIR"' TERM EXIT

# Make conda/mamba available on Metacentrum. run_pipeline.sh switches between
# envs internally (env_dockndesign for QC/selection, env_ligandmpnn for design)
# according to config.yaml.
module add mambaforge

# CPU thread hints for PyTorch / numpy (LigandMPNN auto-picks the visible GPU)
export OMP_NUM_THREADS=${NCPUS}
export MKL_NUM_THREADS=${NCPUS}

echo "=========================================="
echo "Host    : \$(hostname)"
echo "Start   : \$(date)"
echo "Project : ${PROJECT_ROOT}"
echo "Scratch : \$SCRATCHDIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU visible)"
echo "=========================================="

cd "${PROJECT_ROOT}"
bash "${PROJECT_ROOT}/scripts/run_pipeline.sh"${PDB_ARGS}

echo "=========================================="
echo "Done    : \$(date)"
echo "=========================================="
EOF

echo "PBS script written: $RUN_FILE"
echo "Submitting via qsub..."
qsub "$RUN_FILE"

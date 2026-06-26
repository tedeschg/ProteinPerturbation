#!/bin/bash
###############################################################################
# runit_score.sh — Metacentrum PBS wrapper for run_score.sh (PBS job array)
#
# Usage (on a Metacentrum frontend):
#   bash runit_score.sh                # submit as job array (default)
#   bash runit_score.sh --no-array     # submit as a single serial job
#
# Counts PDB files in BACKBONES_DIR, splits them into chunks of CHUNK_SIZE,
# and submits one PBS array slot per chunk. Each slot processes its chunk on
# one GPU. Multiple slots run in parallel up to the queue limits.
#
# To control the chunk size or paths, edit the variables below before
# submitting.
###############################################################################

set -euo pipefail

# =============================================================================
# User-configurable — EDIT THESE for Metacentrum
# =============================================================================
PROJECT_ROOT="/storage/brno12-cerit/home/tedeschg/prj/protein-perturbation"

# Paths that get exported into the job env (override the defaults in run_score.sh)
# Currently reusing Spiwok's installation of LigandMPNN + conda env.
LMPNN_PATH="/storage/brno12-cerit/home/spiwokv/bakeroviny/lmpnn/LigandMPNN"
LMPNN_ENV="/storage/brno12-cerit/home/spiwokv/bakeroviny/lmpnn/my_env"
BACKBONES_DIR="/storage/brno12-cerit/home/tedeschg/prj/protein-perturbation/experiments/dude-z_experiments/reference/output_reference_lmpnn/backbones"
OUT_FOLDER="/storage/brno12-cerit/home/tedeschg/prj/protein-perturbation/experiments/dude-z_experiments/reference/output_reference_score/"

# Job-array configuration
CHUNK_SIZE=50          # PDBs per array slot. Smaller = more parallelism + queue overhead.
                       # Tune so each slot fits comfortably inside WALLTIME.

# PBS resources (per slot)
QUEUE="gpu"
WALLTIME="03:00:00"    # walltime per slot. 50 PDBs × ~30s/PDB on mid-tier GPU ~ 25 min — plenty.
NCPUS=4
NGPUS=1
GPU_MEM="10gb"         # filter out very old/small GPUs (LigandMPNN works fine on 8GB+)
MEM="32gb"
SCRATCH="20gb"
EXCLUDE_CL="cl_bee=False"   # exclude the bee cluster. Leave empty to allow it.

# =============================================================================
# Parse args
# =============================================================================
USE_ARRAY=true
for arg in "$@"; do
    case "$arg" in
        --no-array) USE_ARRAY=false ;;
        -h|--help)
            sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

# =============================================================================
# Count PDBs and compute the array range
# =============================================================================
if [ ! -d "$BACKBONES_DIR" ]; then
    echo "ERROR: BACKBONES_DIR not found: $BACKBONES_DIR" >&2
    echo "Edit the path at the top of $0" >&2
    exit 1
fi

# Compatible with shells that expand non-matching globs to literal "*.pdb"
shopt -s nullglob
PDB_LIST=( "$BACKBONES_DIR"/*.pdb )
shopt -u nullglob
N_PDB=${#PDB_LIST[@]}

if [ "$N_PDB" -eq 0 ]; then
    echo "ERROR: no .pdb files in $BACKBONES_DIR" >&2
    exit 1
fi

N_SLOTS=$(( (N_PDB + CHUNK_SIZE - 1) / CHUNK_SIZE ))
LAST_SLOT=$(( N_SLOTS - 1 ))

echo "Found $N_PDB PDBs in $BACKBONES_DIR"
if $USE_ARRAY; then
    echo "Array: $N_SLOTS slot(s) of up to $CHUNK_SIZE PDB(s) each (#PBS -J 0-${LAST_SLOT})"
else
    echo "Single job: scoring all $N_PDB PDBs serially in one slot"
fi

MYDIR="$(pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_FILE="${MYDIR}/score_${TIMESTAMP}.$$.pbs"

# =============================================================================
# Generate PBS script
# =============================================================================
{
cat <<EOF
#!/bin/sh
#PBS -q ${QUEUE}
#PBS -N score_${TIMESTAMP}
#PBS -l walltime=${WALLTIME}
#PBS -l select=1:ngpus=${NGPUS}:ncpus=${NCPUS}:mem=${MEM}:scratch_local=${SCRATCH}:gpu_mem=${GPU_MEM}${EXCLUDE_CL:+:${EXCLUDE_CL}}
#PBS -o ${MYDIR}/score_${TIMESTAMP}.^array_index^.stdout
#PBS -e ${MYDIR}/score_${TIMESTAMP}.^array_index^.stderr
EOF

if $USE_ARRAY; then
    echo "#PBS -J 0-${LAST_SLOT}"
fi

cat <<EOF

set -euo pipefail

trap 'rm -rf "\$SCRATCHDIR"' TERM EXIT

# Make conda/mamba available
module add mambaforge

# Pass paths + chunk config down to run_score.sh
export LMPNN_PATH="${LMPNN_PATH}"
export LMPNN_ENV="${LMPNN_ENV}"
export BACKBONES_DIR="${BACKBONES_DIR}"
export OUT_FOLDER="${OUT_FOLDER}"
export CHUNK_SIZE=${CHUNK_SIZE}

# CPU thread hints
export OMP_NUM_THREADS=${NCPUS}
export MKL_NUM_THREADS=${NCPUS}

echo "=========================================="
echo "Host        : \$(hostname)"
echo "Start       : \$(date)"
echo "Array slot  : \${PBS_ARRAY_INDEX:-(not array)}"
echo "Total PDBs  : ${N_PDB}"
echo "Chunk size  : ${CHUNK_SIZE}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU visible)"
echo "=========================================="

cd "${PROJECT_ROOT}"
bash "${PROJECT_ROOT}/scripts/run_score.sh"

echo "=========================================="
echo "Done        : \$(date)"
echo "=========================================="
EOF
} > "$RUN_FILE"

echo ""
echo "PBS script written: $RUN_FILE"
echo "Submitting via qsub..."
qsub "$RUN_FILE"

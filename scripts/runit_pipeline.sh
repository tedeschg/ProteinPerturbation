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

# gpu_cap only lets you request a MINIMUM compute capability on Metacentrum
# (there is no upper-bound syntax), so it cannot be used to exclude newer
# GPUs. The current torch build in my_env only supports up to sm_90, and the
# RTX PRO 6000 Blackwell (sm_120) cards live on the "grogu" cluster
# (grogu[1-3].cerit-sc.cz, CERIT-SC). So we exclude that cluster explicitly.
# If other Blackwell/unsupported-arch clusters show up later, add them here
# too (comma is not supported for combining cl_X=False conditions on the same
# resource name, but cl_NAME are separate resources, so you CAN stack them):
#   EXCLUDE_CLUSTERS="cl_grogu=False:cl_othercluster=False"
EXCLUDE_CLUSTERS="cl_grogu=False"

# Minimum required architecture actually supported by my_env's torch build.
# This won't exclude sm_120 (since gpu_cap has no upper bound), but it stops
# the job landing on genuinely too-old GPUs that torch also doesn't support.
GPU_CAP_MIN="sm_50"

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
#PBS -l select=1:ngpus=${NGPUS}:ncpus=${NCPUS}:mem=${MEM}:scratch_local=${SCRATCH}:gpu_mem=${GPU_MEM}:gpu_cap=${GPU_CAP_MIN}${EXCLUDE_CLUSTERS:+:${EXCLUDE_CLUSTERS}}
#PBS -o ${MYDIR}/pipeline_${TIMESTAMP}.stdout
#PBS -e ${MYDIR}/pipeline_${TIMESTAMP}.stderr

set -euo pipefail

# Auto-clean scratch on exit. "|| true" so a stray permission issue on some
# leftover file never masks the job's real exit status or crashes the trap.
trap 'rm -rf "\$SCRATCHDIR" 2>/dev/null || true' TERM EXIT

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
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader 2>/dev/null || echo "(no GPU visible)"
echo "=========================================="

# --- Runtime safety net -----------------------------------------------------
# Even with the PBS-side exclusions above, fail fast and CLEARLY if we ever
# land on a GPU whose compute capability the installed torch build doesn't
# support, instead of dying deep inside score.py with an opaque CUDA error.
# Adjust MAX_SUPPORTED_CC if/when my_env's torch is upgraded (check with:
#   python -c "import torch; print(torch.cuda.get_arch_list())"
# ).
MAX_SUPPORTED_CC="9.0"
GPU_CC="\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1)"
if [ -n "\$GPU_CC" ]; then
    if awk -v cc="\$GPU_CC" -v max="\$MAX_SUPPORTED_CC" 'BEGIN { exit !(cc > max) }'; then
        echo "ERROR: GPU compute capability \$GPU_CC exceeds what the installed" >&2
        echo "       PyTorch build supports (max sm_\$(echo \$MAX_SUPPORTED_CC | tr -d .))." >&2
        echo "       This node's GPU is likely an RTX PRO 6000 Blackwell (sm_120)." >&2
        echo "       Refusing to run to avoid a confusing crash mid-script." >&2
        echo "       Fix: upgrade torch in my_env, or check that cluster exclusion" >&2
        echo "       (EXCLUDE_CLUSTERS) in runit_pipeline.sh is still up to date." >&2
        exit 1
    fi
fi
# -----------------------------------------------------------------------------

cd "${PROJECT_ROOT}"
bash "${PROJECT_ROOT}/scripts/run_pipeline.sh"${PDB_ARGS}

echo "=========================================="
echo "Done    : \$(date)"
echo "=========================================="
EOF

echo "PBS script written: $RUN_FILE"
echo "Submitting via qsub..."
qsub "$RUN_FILE"
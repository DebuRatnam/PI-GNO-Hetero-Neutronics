#!/bin/bash
# SLURM array job for sharded dataset generation. CPU ONLY -- no GPU is used or
# requested anywhere in this file. Sample generation is an eigenvalue solve per
# core, so it parallelizes perfectly across array tasks.
#
#   sbatch --array=0-39 slurm_generate.sbatch bc   ../../datasets/hex_bc
#   sbatch --array=0-29 slurm_generate.sbatch res  ../../datasets/hex_probe
#
# then ONCE, after the array finishes:
#   python sharding.py ../../datasets/hex_bc --expected 3000
#
# The merge is deliberately a separate step: a task writing straight into
# manifest.json would race every other task and silently drop rows.
#
# ---- FILL THESE IN for your cluster -----------------------------------------
#SBATCH --job-name=pigno-gen
#SBATCH --account=CHANGEME              # e.g. fc_<project>
#SBATCH --partition=CHANGEME            # a CPU partition, NOT a GPU one
#SBATCH --qos=CHANGEME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2               # scipy splu is largely single-threaded;
                                        # more array tasks beats more threads
#SBATCH --time=04:00:00
#SBATCH --output=logs/gen_%A_%a.out
#SBATCH --error=logs/gen_%A_%a.out
# -----------------------------------------------------------------------------

set -euo pipefail
MODE="${1:?usage: sbatch slurm_generate.sbatch <bc|res> <out_dir> [extra args]}"
OUT="${2:?missing output directory}"
shift 2

mkdir -p logs

# The data-generation environment: OpenMC + numpy/scipy, and NO torch. Keep it
# separate from the benchmark env -- xs_openmc must never import torch.
PY="${PIGNO_DATAGEN_PYTHON:-python}"

N="${SLURM_ARRAY_TASK_COUNT:-1}"
I="${SLURM_ARRAY_TASK_ID:-0}"
echo "shard $I of $N  mode=$MODE  out=$OUT  host=$(hostname)  $(date)"

# One thread per solve. Oversubscribing BLAS inside an array job makes every
# task slower, because the tasks are already competing for the same cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

case "$MODE" in
  bc)
    "$PY" -u generate_bc.py --out "$OUT" --shard "$I" --n-shards "$N" "$@"
    ;;
  res)
    "$PY" -u generate_resolution.py --out "$OUT" --shard "$I" --n-shards "$N" "$@"
    ;;
  *)
    echo "unknown mode '$MODE' (expected bc or res)" >&2; exit 2
    ;;
esac

echo "shard $I done  $(date)"

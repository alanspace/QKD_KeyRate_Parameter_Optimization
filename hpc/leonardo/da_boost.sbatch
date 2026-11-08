#!/bin/bash
#SBATCH --job-name=da_boost
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=da_boost_%j.out
#  For accounts without DCGP access (Booster-only): run the DA chunks on the 32 CPU cores of one Booster node, no GPU.
#  Resumable: chunks whose file exists are skipped, so resubmit until all NCHUNKS files exist, then run da_merge
#  (on boost: sbatch --account=<ACCOUNT> -p boost_usr_prod --qos=boost_qos_dbg hpc/leonardo/da_merge.sbatch).
#  submit with: sbatch --account=<ACCOUNT> hpc/leonardo/da_boost.sbatch   (drop --qos line for the regular queue)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f Optimization/benchmark/da_dataset.py ] || { echo "submit from the main-branch repo root"; exit 1; }
module load python/3.11.7 >/dev/null 2>&1 || true
PRESET=${PRESET:-lim}
NCHUNKS=${NCHUNKS:-100}
export JAX_PLATFORMS=cpu PYTHONPATH="$PWD" PRESET NCHUNKS
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs_da
seq 0 $((NCHUNKS - 1)) | xargs -P "$SLURM_CPUS_PER_TASK" -I{} bash -c '
  c=$(printf "%04d" {}); n=$(printf "%04d" "$NCHUNKS")
  [ -f "results/da_dataset_${PRESET}/chunk_${c}_of_${n}.json" ] && exit 0
  .venv/bin/python Optimization/benchmark/da_dataset.py run --preset "$PRESET" --chunk {} --nchunks "$NCHUNKS" > "logs_da/chunk_${c}.log" 2>&1 \
    || echo "chunk {} failed or was cut off; resubmit to redo it"'
echo "done chunks: $(ls results/da_dataset_${PRESET}/chunk_*_of_$(printf %04d $NCHUNKS).json 2>/dev/null | wc -l) / $NCHUNKS"

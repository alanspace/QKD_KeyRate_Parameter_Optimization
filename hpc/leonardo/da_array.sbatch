#!/bin/bash
#SBATCH --job-name=da_dataset
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --array=0-99
#SBATCH --output=da_%A_%a.out
#  submit with: sbatch --account=<ACCOUNT> hpc/leonardo/da_array.sbatch
set -euo pipefail
cd "$WORK/qkd"
PRESET=lim
NCHUNKS=100
export JAX_PLATFORMS=cpu PYTHONPATH="$PWD"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # one core per task
.venv/bin/python Optimization/benchmark/da_dataset.py run --preset "$PRESET" --chunk "$SLURM_ARRAY_TASK_ID" --nchunks "$NCHUNKS"

#!/bin/bash
#SBATCH --job-name=da_merge
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=da_merge_%j.out
set -euo pipefail
cd "$WORK/qkd"
export JAX_PLATFORMS=cpu PYTHONPATH="$PWD"
.venv/bin/python Optimization/benchmark/da_dataset.py merge --preset lim

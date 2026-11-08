#!/bin/bash
#SBATCH --job-name=da_merge
#SBATCH --partition=dcgp_usr_prod   # Booster-only accounts: sbatch -p boost_usr_prod [--qos=boost_qos_dbg]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=da_merge_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"   # submit from the repo root (clone may be in $WORK or scratch)
[ -f Optimization/benchmark/da_dataset.py ] || { echo "submit from the main-branch repo root"; exit 1; }
module load python/3.11.7 >/dev/null 2>&1 || true
export JAX_PLATFORMS=cpu PYTHONPATH="$PWD"   # replaces the module's PYTHONPATH
.venv/bin/python Optimization/benchmark/da_dataset.py merge --preset lim

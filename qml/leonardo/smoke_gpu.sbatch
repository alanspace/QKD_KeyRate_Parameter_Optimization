#!/bin/bash
#SBATCH --job-name=qml_smoke
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=qml_smoke_%j.out
#  Smoke test before the final array (2026-11-08, job 63910045): one dense seed at E=100 on one A100; prints the GPU,
#  the JAX devices and float64. Submit from the repo root: sbatch --account=<ACCOUNT> qml/leonardo/smoke_gpu.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python/3.11.7 >/dev/null 2>&1 || true
export PYTHONPATH="$PWD"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
.venv-qml/bin/python -c "import jax; jax.config.update(\"jax_enable_x64\", True); print(\"devices\", jax.devices(), \"x64\", jax.numpy.ones(1).dtype)"
QML_E=100 .venv-qml/bin/python qml/train.py final --variants dense --seeds 1

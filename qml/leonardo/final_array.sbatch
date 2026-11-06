#!/bin/bash
#SBATCH --job-name=qml_final
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=0-8
#SBATCH --output=qml_final_%A_%a.out
#  --account is set at submission: sbatch --account="$LEO_ACCOUNT" qml/leonardo/final_array.sbatch
# Task map: 0-3 -> dense seeds 1-4; 4-8 -> sparse seeds 0-4. Each task writes its own parts (no shared files).
set -euo pipefail
cd "$WORK/qkd_qml"
VARS=(dense dense dense dense sparse sparse sparse sparse sparse)
SEEDS=(1 2 3 4 0 1 2 3 4)
V=${VARS[$SLURM_ARRAY_TASK_ID]}; S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
export PYTHONPATH="$PWD"
.venv-qml/bin/python -c "import jax; jax.config.update('jax_enable_x64', True); print('devices', jax.devices(), 'x64', jax.numpy.ones(1).dtype)"
.venv-qml/bin/python qml/train.py final --variants "$V" --seeds "$S"

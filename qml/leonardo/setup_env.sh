#!/usr/bin/env bash
# On the Leonardo login node (internet access): clone branch qml-surrogate from GitHub (public) into $WORK/qkd_qml
# and create .venv-qml (CUDA JAX 0.10.2 + PennyLane 0.45.1, float64). The finished seed-0 parts are in the repo.
set -euo pipefail
cd "$WORK"
if [ -d qkd_qml/.git ]; then git -C qkd_qml pull --ff-only; else
  git clone -b qml-surrogate https://github.com/alanspace/QKD_KeyRate_Parameter_Optimization.git qkd_qml; fi
cd qkd_qml
module load python 2>/dev/null || true
python3 -m venv .venv-qml
.venv-qml/bin/pip install -q --upgrade pip
.venv-qml/bin/pip install -q "jax[cuda12]==0.10.2" "pennylane==0.45.1" numpy scipy
.venv-qml/bin/python -c "import jax, pennylane; print('jax', jax.__version__, 'pennylane', pennylane.__version__)"
ls qml/data/final_parts | wc -l    # expect 5 (dense seed 0) before the GPU run

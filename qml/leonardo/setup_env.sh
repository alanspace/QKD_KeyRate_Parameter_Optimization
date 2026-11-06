#!/usr/bin/env bash
# On the Leonardo login node (internet access). Creates $WORK/qkd_qml with .venv-qml (CUDA JAX, float64).
set -euo pipefail
cd "$WORK"
[ -d qkd_qml ] || git clone -b qml-surrogate qkd_qml_in/qkd_qml.bundle qkd_qml
cd qkd_qml && git pull --ff-only ../qkd_qml_in/qkd_qml.bundle qml-surrogate || true
mkdir -p qml/data/final_parts && cp -n ../qkd_qml_in/final_parts/*.npz qml/data/final_parts/ || true
module load python 2>/dev/null || true
python3 -m venv .venv-qml
.venv-qml/bin/pip install -q --upgrade pip
.venv-qml/bin/pip install -q "jax[cuda12]==0.10.2" "pennylane==0.45.1" "numpy" "scipy"
.venv-qml/bin/python -c "import jax, pennylane; print('jax', jax.__version__, 'pennylane', pennylane.__version__)"
echo "env ready; GPU visibility is checked inside the job"

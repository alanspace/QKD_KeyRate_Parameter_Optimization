#!/usr/bin/env bash
# Leonardo -> Mac: copy the finished parts and job logs back.
set -euo pipefail
cd "$(dirname "$0")/../.."
source qml/leonardo/env.sh
scp "$LEO_USER@$LEO_HOST:\$WORK/qkd_qml/qml/data/final_parts/*.npz" qml/data/final_parts/
mkdir -p results/logs_qml/leonardo && scp "$LEO_USER@$LEO_HOST:\$WORK/qkd_qml/qml_final_*.out" results/logs_qml/leonardo/ || true
ls qml/data/final_parts | wc -l

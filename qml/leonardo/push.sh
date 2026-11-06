#!/usr/bin/env bash
# Mac -> Leonardo: git bundle of the committed branch (provenance stamps keep working) + finished parts.
set -euo pipefail
cd "$(dirname "$0")/../.."
source qml/leonardo/env.sh
git diff --quiet HEAD -- qml/train.py qml/models.py qml/common.py || { echo "commit qml code first"; exit 1; }
git bundle create /tmp/qkd_qml.bundle qml-surrogate
ssh "$LEO_USER@$LEO_HOST" 'mkdir -p $WORK/qkd_qml_in'
scp /tmp/qkd_qml.bundle "$LEO_USER@$LEO_HOST:\$WORK/qkd_qml_in/"
scp -r qml/data/final_parts "$LEO_USER@$LEO_HOST:\$WORK/qkd_qml_in/"
echo "pushed $(git rev-parse --short HEAD)"

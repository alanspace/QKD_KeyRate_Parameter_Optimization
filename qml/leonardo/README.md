# Final training on CINECA Leonardo (A100 GPUs)

Resume support: `qml/train.py final` checkpoints every (variant, model, seed) as a part in `qml/data/final_parts/`
and skips parts that exist. Dense seed 0 comes from the stopped CPU run (`--import-cpu-s0`). The remaining 45 parts are:
- dense seeds 1-4;
- sparse seeds 0-4.

Each array task trains one (variant, seed) on one A100, in float64. Same code, data, steps and seeds as preregistered;
see Amendment 5.

Fill in `LEO_USER` and `LEO_ACCOUNT` in `env.sh` first. The SSH certificate is renewed by the owner.

1. On the Leonardo login node: `bash setup_env.sh`. Get the script with
   `curl -O https://raw.githubusercontent.com/alanspace/QKD_KeyRate_Parameter_Optimization/qml-surrogate/qml/leonardo/setup_env.sh`,
   or copy it. It clones `qml-surrogate` from GitHub into `$WORK/qkd_qml` and builds `.venv-qml`.
2. `cd $WORK/qkd_qml && sbatch --account=<ACCOUNT> qml/leonardo/final_array.sbatch` (9 tasks).
3. `bash qml/leonardo/pull.sh` on the Mac copies the parts back. Alternatively, commit and push them from Leonardo,
   if a GitHub credential is configured there.
   Then locally: `train.py final --assemble`, run_qml steps `latency` and `certify_final`, then Amendment 4.

Nothing here is submitted without the owner's approval of the allocation use.

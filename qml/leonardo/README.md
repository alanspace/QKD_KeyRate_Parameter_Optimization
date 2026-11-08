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
   or copy it. It clones `qml-surrogate` from GitHub into `${QKD_PARENT:-$WORK}/qkd_qml` and builds `.venv-qml`.
2. From the repo root (jobs run from the submit directory): `sbatch --account=<ACCOUNT> qml/leonardo/final_array.sbatch` (9 tasks).
   **When the regular Booster queue is full** (start estimates of days): use the debug QOS (30 min per job, at most
   4 queued and 2 running per user). One (variant, seed) takes about 10-15 min on an A100 (smoke test 2026-11-08: PQC-full
   8x6 at E=100 took 43 s including compilation). Submit in batches of up to 4 tasks:
   `sbatch --account=<ACCOUNT> --qos=boost_qos_dbg --time=00:30:00 --array=0-3%2 qml/leonardo/final_array.sbatch`, then
   `--array=4-7%2`, then `--array=8`. A task cut off at 30 min resumes from its finished parts when resubmitted.
3. `bash qml/leonardo/pull.sh` on the Mac copies the parts back. Alternatively, commit and push them from Leonardo,
   if a GitHub credential is configured there.
   Then locally: `train.py final --assemble`, run_qml steps `latency` and `certify_final`, then Amendment 4.

Nothing here is submitted without the owner's approval of the allocation use.

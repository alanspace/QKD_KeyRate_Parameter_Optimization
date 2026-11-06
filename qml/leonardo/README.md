# Final training on CINECA Leonardo (A100 GPUs)

Resume support: `qml/train.py final` checkpoints every (variant, model, seed) as a part in `qml/data/final_parts/`
and skips parts that exist. Dense seed 0 comes from the stopped CPU run (`--import-cpu-s0`). The remaining 45 parts are:
- dense seeds 1-4;
- sparse seeds 0-4.

Each array task trains one (variant, seed) on one A100, in float64. Same code, data, steps and seeds as preregistered;
see Amendment 5.

Fill in `LEO_USER` and `LEO_ACCOUNT` in `env.sh` first. The SSH certificate is renewed by the owner.

1. `bash qml/leonardo/push.sh` (on the Mac): git bundle of branch qml-surrogate + the finished parts → Leonardo.
2. `bash qml/leonardo/setup_env.sh` (on the Leonardo login node; compute nodes have no internet):
   clone from the bundle, create `.venv-qml` with jax[cuda12]==0.10.2 and pennylane==0.45.1.
3. `sbatch qml/leonardo/final_array.sbatch` (on Leonardo): 9 tasks.
4. `bash qml/leonardo/pull.sh` (on the Mac): copy the parts back.
   Then, locally: `train.py final --assemble`, then run_qml steps `latency`, `certify_final`, and Amendment 4.

Nothing here is submitted without the owner's approval of the allocation use.

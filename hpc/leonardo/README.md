# Running this repository on CINECA Leonardo

The repository is public, so Leonardo pulls code directly from GitHub. Login nodes have internet; compute nodes do
not, so environments are built on the login node.

**Prerequisites from the owner:**
- a valid MyAccessID SSH certificate (renewed interactively, about 12 h validity);
- the username;
- the SLURM project account, and approval to spend the allocation.

## 1. Get the code (login node)

```bash
cd $WORK          # or any folder, e.g. /leonardo_scratch/large/userexternal/<user>/WORK; jobs run from where you sbatch
git clone https://github.com/alanspace/QKD_KeyRate_Parameter_Optimization.git qkd   # first time
cd qkd && git pull                                                                # later: update to the latest main
```

## 2. Physics environment (CPU; float64)

```bash
bash hpc/leonardo/setup_env_main.sh        # creates .venv (numpy, scipy, jax[cpu]==0.9.0) and runs a quick self-test
```

## 3. Dual-annealing dataset (CPU partition): `Optimization/benchmark/da_dataset.py`

This regenerates the 6,000-point DA dataset with the archived DA pipeline (thesis grid) and scores it against the
verified optimizer.

```bash
sbatch --account=<ACCOUNT> hpc/leonardo/da_array.sbatch          # 100 tasks × 1 core; preset lim (edit PRESET for legacy)
# when all tasks are done:
sbatch --account=<ACCOUNT> --dependency=afterok:<JOBID> hpc/leonardo/da_merge.sbatch
```

**Booster-only accounts** (no DCGP access, e.g. AIFAC_P02_1031): use `hpc/leonardo/da_boost.sbatch` instead: one
Booster node, 32 chunks in parallel on its CPU cores, debug QOS (30 min). It skips finished chunks, so resubmit until
all 100 exist, then `sbatch --account=<ACCOUNT> -p boost_usr_prod --qos=boost_qos_dbg hpc/leonardo/da_merge.sbatch`.

Output: `results/da_dataset_<preset>/` (chunks, `da_dataset.json`, `DA_DATASET.md`).

Estimated cost: about 3,000 DA points × 6–18 s + 3,000 fast L-BFGS-B points, i.e. ~10 CPU-hours in total. Measured on
an M2 Pro (110 km point: 5.5 s); Leonardo cores will differ.

## 4. QML surrogate training (GPU partition)

This lives on branch `qml-surrogate`; see `qml/leonardo/README.md` and `qml/HANDOFF.md` there.
`git checkout qml-surrogate` in a separate clone (e.g. `$WORK/qkd_qml`).

## 5. Getting results back

Choose one:
- `scp -r <user>@login.leonardo.cineca.it:'$WORK/qkd/results/da_dataset_lim' results/` on the Mac, then commit there;
- commit and push from Leonardo. This needs a GitHub token or SSH key configured on Leonardo by the owner.

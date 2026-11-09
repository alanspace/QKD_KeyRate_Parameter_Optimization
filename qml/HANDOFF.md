# Handoff: QML surrogate study, remaining work (state at 2026-11-06 ~14:50 CET)

This file is for whoever finishes the study, human or AI. Read it fully before running anything.

## 0. Where things are

| item | value |
|---|---|
| Worktree | `QBAI_Code_Workspace/QKD/QKD_qml_surrogate_wt` (git worktree of `QKD/QKD_KeyRate_Parameter_Optimization`) |
| Branch | `qml-surrogate` on GitHub (`origin/qml-surrogate`). Work on this branch. **Do not merge it into `main` without the owner's approval**; the study is unfinished. |
| Main `.venv` | Physics, JAX 0.9.0, float64, CPU. Symlinked from the main checkout: `.venv -> ../QKD_KeyRate_Parameter_Optimization/.venv` |
| `.venv-qml` | Models: PennyLane 0.45.1, JAX **0.10.2**. jax 0.11 / 0.12 break PennyLane at runtime. No physics import is allowed in this env. |
| Run everything with | `JAX_PLATFORMS=cpu PYTHONPATH=.` from the worktree root. `qml/run_qml.py` sets both itself. |
| Protocol | `result_report/final_report.tex` Section 7 (binding) + `qml/PREREGISTRATION.md` with Amendments 1–5 |
| Log | `STATUS.md` sections 10–10e (measured numbers only) |
| Step records | `results/MANIFEST_qml.json`, logs in `results/logs_qml/` |

**Setup from GitHub (fresh machine):**
```bash
git clone https://github.com/alanspace/QKD_KeyRate_Parameter_Optimization.git && cd QKD_KeyRate_Parameter_Optimization
git checkout qml-surrogate
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt "jax[cpu]==0.9.0"   # physics env (main .venv)
python3.13 -m venv .venv-qml && .venv-qml/bin/pip install -r qml/requirements-qml.txt      # models env (jax 0.10.2)
JAX_PLATFORMS=cpu .venv/bin/python -m unittest tests.test_qml.test_qml_main                     # sanity check
```
The original machine used a git worktree and symlinked `.venv` from the main checkout. A plain clone with the two
environments above is equivalent.

## 1. Rules that bind every remaining step

1. **No model-predicted key rate is ever reported.** Every R is recomputed by `qml/certify.py` with the float64 `lim`
   model at the predicted parameters.
2. **Predictions are u-vectors mapped through `src/qkd_opt/reparam.py`.** They are feasible by construction.
3. **Code you may not modify:** `src/qkd_opt/`, `src/qkd/`, `Optimization/` and the existing tests. New code goes only
   under `qml/` and `tests/test_qml/`.
4. **Protocol changes** are made only as dated amendments appended to `qml/PREREGISTRATION.md`. Never edit earlier
   text.
5. **IQM hardware:**
   - Nothing is submitted without the owner's written "GO HARDWARE", given after they have seen the backend, circuits,
     shots and estimated credits.
   - The token comes only from the env var `IQM_TOKEN`; never print, log or commit it.
6. **STATUS.md:** a dated section per step, with measured numbers only.
7. **Commits:** commit on `qml-surrogate` with the attribution line used so far. Pushing that branch is fine. **Never merge into `main` or push to `main`** without the owner.
8. **Presentation language:** no "advantage", "breakthrough" or "proves". A negative result is valid. The preliminary
   validation numbers already point to the classical models winning (section 6).

## 2. What is done

| step | status | output |
|---|---|---|
| Labels (61,875; quality gate PASS) | done | `qml/data/labels_0d8bef1402_9fa16da52f39.{npz,json}` |
| Pilot | done | `results/qml_pilot_summary_d118356a45_d36bcf048d7d/PILOT.md` |
| Tune (6 configs × 3 families + ridge, seed 0) | done, 4814 s | `results/qml_tune_*/{tune.json,predictions_val.npz}` |
| Select | done | `results/qml_select_7218993cb5_feab976f711d/selection.json` |
| Final | **paused**: 5 of 50 parts done | `qml/data/final_parts/dense__{PQC-full,PQC-product,NN-tuned,NN-matched,RFF-ridge}__s0.npz` |
| Latency, certify_final, repair, report, numbers | not run | — |
| IQM prep | rebuild check PASS on random weights; transpile plan done | `qml/iqm/rebuild.py` |

The selection picked:
- **PQC-full:** 8 qubits × 6 layers (240 params);
- **PQC-product:** 6×6;
- **NN:** h = 12 (257 params). NN-tuned and NN-matched are therefore **identical** in this run;
- **RFF-ridge:** the 5×6 spectrum, so each seed trains an extra PQC-full 5×6 for its frequencies.

## 3. Remaining steps, in order

### Step A: final training on CINECA Leonardo (A100), Amendment 5

Prerequisites from the owner:
- a renewed MyAccessID SSH certificate (`~/.ssh/id_login.leonardo.cineca.it-cert.pub`; it expired 2026-11-04);
- the username and SLURM account, filled into `qml/leonardo/env.sh`;
- **approval to use the allocation.**

```bash
# on the Leonardo login node (the repo is public; no bundle or credentials needed to read it)
curl -O https://raw.githubusercontent.com/alanspace/QKD_KeyRate_Parameter_Optimization/qml-surrogate/qml/leonardo/setup_env.sh
bash setup_env.sh                              # clones qml-surrogate into $WORK/qkd_qml, builds .venv-qml (CUDA)
cd <clone>/qkd_qml && sbatch --account=$LEO_ACCOUNT qml/leonardo/final_array.sbatch   # 9 tasks: dense s1-4, sparse s0-4
squeue -u $USER                                # monitor; each task prints its devices and that float64 is in use
bash qml/leonardo/pull.sh                      # Mac: copy the parts back (expect 50 files), or push them from Leonardo
```

Checks:
- Every job log shows a CUDA device and `float64`.
- `ls qml/data/final_parts | wc -l` = 50.
- Each part's `meta.device.backend` is `gpu`, except the 5 dense s0 parts, which are `cpu`.

Notes:
- `setup_env.sh` assumes `module load python` gives Python ≥ 3.11. Adjust the module name if Leonardo differs.
- Compute nodes have no internet, so build the environment on the login node.
- If a task dies, resubmit only that array index. Existing parts are skipped.

**Optional (Amendment 5): cross-hardware check.** Clone again into a scratch directory with an empty `final_parts/`.
Run `train.py final --variants dense --seeds 0` on a GPU. Compare its loss and predictions with the CPU parts, and
report max |Δu| and Δloss. The CPU parts stay the official seed 0.

### Step B: assemble, latency, certification (Mac, main + qml envs)

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. .venv-qml/bin/python qml/train.py final --assemble    # writes results/qml_final_*/
.venv/bin/python qml/run_qml.py --only latency certify_final                          # run on AC power
```

- `certify_final` takes about 25–40 min. It runs batched cold solves at about 21k evaluation points, plus hybrid
  k = 1..3 for 54 prediction sets.
- `--assemble` also rewrites `qml/data/weights_*_dense_s0.npz`; latency and IQM read these.
- Do not run anything heavy in parallel with `latency`, because it is a timing measurement.

### Step C: label repair + re-certification (Amendment 4)

```bash
.venv/bin/python qml/repair_labels.py          # on AC, about 10 min; writes qml/data/repaired/labels_repaired_*.npz
.venv/bin/python qml/run_qml.py --only certify_final --extra --labels qml/data/repaired/labels_repaired_<git>_<cfg>.npz
```

Acceptance:
- `gate_pass: true` in `results/qml_repair_*/repair_summary.json`.
- **Known sub-optimal labels to check, status "to be verified".** The repair is expected to fix these, but this is
  NOT confirmed until the repair has run and these exact points have been re-checked against cold solves.
  At least **77** sub-optimal labels are known, max relative deficit **7.5e-4**:
  - 58 just before 10 silent switches (STATUS 10d; up to 6.0e-4);
  - 19 at n_X = 1e7, L = 105.375–105.938 km (STATUS 11(c); up to 7.47e-4; split train 1, val 1, test 3, unused 14).
    There the forward sweep logged the switch, but its backfill stopped about 0.55 km too early. Cold solves put the
    crossover at 105.35–105.40 km.

  After `repair_labels.py`:
  - re-run cold solves (`optimize_batch`, 2^14 Sobol, top-16) at all 77 points, plus the window around each;
  - confirm that every repaired label is within 1e-9 of the cold solve;
  - record the per-point result in STATUS.

  Only then call them fixed. If any point is still sub-optimal, stop and report.
- The repaired crossover list should also contain the cold-solve-confirmed switches in `final_report.tex` Table
  "Below-horizon branch switches" (at the six reference n_X). Report any that are missing.
- Record `n_repaired`, the change in the crossover-window list, and the new crossover count.

**Code task: the report must show both certifications.** `qml/report.py` currently reads only the newest
`qml_certify_final_*`. Extend it:
- the primary tables use the certification against the **repaired** labels;
- add a secondary table "against original labels";
- the groups and crossover windows come from the repaired label file.

### Step D: report and macros

```bash
.venv/bin/python qml/run_qml.py --only report numbers
```

**`qml/report.py` and `qml/make_numbers_qml.py` have never run on real outputs.** Expect small bugs; fix them in
`qml/` only. Then review `qml/RESULTS.md` against these checks:
- **Kill criterion:** PQC-full median bulk **test** regret (mean over 5 seeds) above 1e-2 means a negative result,
  with no retuning. The preliminary validation value was 5.4e-3 pooled and 2.8e-3 bulk.
- **Mean ± sd over seeds.** The lookups are deterministic.
- **Supplementary score.** The `supp_ref_incl_cliff_*` rows (the 7 reference cliff_edge points, which are ordinary
  optima per STATUS 10a) sit next to the registered `ref_*` rows.
- **Neighbour distance note.** Scenario (ii): bulk 0.25–0.375 km, tail 0.0625–0.094 km, the previous lattice point for
  heldout/extrap. Put it next to the preregistration's original "0.5 / 0.125 km" text, with the reason (owner decision
  (a)).
- **Usefulness verdicts.** They are post hoc and must be labelled as such. Raw numbers sit beside them.
- **NN-tuned = NN-matched.** State this once, and don't count it as two results.
- **Training time per hardware.** CPU seed 0 vs GPU seeds, never pooled.
- **The sentence "No quantum advantage is claimed."** It must be present.
- Fourier spectrum, sparse-variant table, hybrid tables, latency table, shot estimate.

### Step E: slides and transcript

```bash
.venv/bin/python qml/slides/make_showcase.py                       # draws assets/showcase_predictions.png
.venv/bin/python qml/slides/make_params_grid.py --labels qml/data/repaired/labels_repaired_<...>.npz
```

**`qml/slides/pitch.html`:**
- **Slide 10, scoreboard.** Replace the "preliminary" values (validation, seed 0) with the final **test** values,
  mean over 5 seeds, against the repaired labels.
  - Fill "time per answer" from `results/qml_latency_*/latency.json` (batch-1 end-to-end).
  - Fill "quantum + fix-up" as surrogate latency + k = 1 Newton (about 0.9 ms) and its no-key %.
  - Remove the word "preliminary".
- **Slide 11, showcase.** Insert `assets/showcase_predictions.png`.
- **Slide 9.** Fill in the NN training time; the quantum training time is per hardware.
- **Appendix.** Update the sources in the notes slide.

**`qml/slides/TRANSCRIPT.md`:** resolve every **PENDING** line with the true version.

**`qml/slides/slides.html`** (detailed deck): fill the three "pending" result slides and the comparison table from
RESULTS.md.

Re-render each changed slide with headless Chrome and look at it. Check for overlaps; this caught real bugs before.

### Step F: close out

```bash
.venv/bin/python -m unittest tests.test_qml.test_qml_main
PYTHONPATH=. .venv-qml/bin/python -m unittest tests.test_qml.test_qml_models tests.test_qml.test_qiskit_rebuild
JAX_PLATFORMS=cpu PYTHONPATH=. .venv-qml/bin/python qml/iqm/rebuild.py --check     # Addition C on the TRAINED weights
.venv/bin/python -m unittest discover tests                                       # CI-equivalent; about 7 min
```

- Write STATUS section 10f: the measured numbers of steps A–E.
- Commit on `qml-surrogate` and push the branch. Do not merge into `main`.
- Report to the owner at **STOP 3**:
  - the RESULTS summary;
  - the kill-criterion outcome;
  - the usefulness verdicts;
  - the open items.

### Step H: dual-annealing dataset on Leonardo CPUs (main-project task; code on `main`, merged here)

`Optimization/benchmark/da_dataset.py` and `hpc/leonardo/{setup_env_main.sh,da_array.sbatch,da_merge.sbatch}`; see
`hpc/leonardo/README.md`.

- It regenerates the 6,000-point thesis-grid dataset with the archived DA pipeline and scores it against the verified
  optimizer.
- Cost: about 10 CPU-hours, as 100 array tasks of 1 core.
- Default preset `lim`. The thesis's own physics (e_mis 5e-3, p_ap 0, fixed eps) is not a defined preset.
- Smoke test on the Mac: a 0.4 km point (L-BFGS-B branch) gave R < 0 where the verified optimizer finds a key.
- The output goes into STATUS (new section) and resolves Appendix A's "to be regenerated on cloud compute" in
  `final_report.tex`. The report text itself is edited only on `main`, with the owner's approval.

### Step G (only with the owner's approval): Phase 5a IQM inference demo

- `qml/iqm/rebuild.py --check` (must be ≤ 1e-10 on the trained weights) and `--plan`.
- The fake-backend run (`--fake N S`) is **documented but not implemented**: implement it in `qml/iqm/rebuild.py`
  first.
- Expect the 8×6 circuit on Garnet to need extra CZ gates for routing: a 5- or 8-qubit ring on a square lattice. The
  random-weight plan for 5×6 gave 50 CZ vs the ideal 30.
- Show the owner the backend, circuits, shots and the credit estimate (the owner supplies the pricing). Wait for
  "GO HARDWARE".

## 4. Pitfalls already met

- **"Latest file" globbing.** `train.py`, `certify.py` and `report.py` pick the newest `results/qml_*` directory.
  Delete smoke-test outputs before real runs. Use `--labels` explicitly for repaired labels.
- **Labels default.** `certify.py` defaults to the newest `qml/data/labels_*.npz`, which is the original labels. The
  repaired labels live in `qml/data/repaired/` precisely so that nothing picks them up by accident.
- **Long-running steps.** Never start a second heavy job alongside a timing step. Headless Chrome occasionally hangs;
  use `timeout 60`.
- **Mac GPU.** jax-metal supports neither complex64 nor float64, so circuit simulation can't use the Mac GPU.
- **Reference flags are wrong.** The reference `cliff_edge` flags are a normalisation artefact (STATUS 10a). In this
  study the registered rule is kept. On main, `final_report.tex` §4.1, the "two switches" statement (§5.3) and open
  issue 1 still need correcting. That is outside this branch's scope; list it, don't edit main from here.
- **Provenance.** A stopped run's stamp may say "uncommitted changes in qml/". Commit code before every heavy step.

## 5. Expected final artefacts

- **Outputs:** `results/qml_final_*`, `qml_latency_*`, two `qml_certify_final_*` (original and repaired labels),
  `qml_repair_*`.
- **Documents:** `qml/RESULTS.md`, `result_report/generated_numbers_qml.tex`, `results/NUMBERS_qml.md`.
- **Slides:** updated `pitch.html`, `slides.html`, `TRANSCRIPT.md` and assets.
- **Logs:** STATUS 10f, MANIFEST records for every step.

## 6. Preliminary numbers to sanity-check against (validation, seed 0; not results)

Median key lost (regret), bulk and tail pooled:

| model | key lost |
|---|---|
| NN h=12 | 9.1e-4 |
| PQC-full 8×6 | 5.4e-3 |
| RFF ridge | 1.3e-2 |
| PQC-product 6×6 | 2.0e-2 |

Near the maximum distance, share of links left with no key:

| model | no key |
|---|---|
| lookup-2D | 0.8 % |
| NN | 9.4 % |
| PQC-full | 11.9 % |

PQC-full + 1 Newton step: median 99.998 % (typical distance) / 99.71 % (near the edge) of the optimal key.

Final test numbers should be of the same order. If they differ by orders of magnitude, suspect a pipeline bug before
reporting.

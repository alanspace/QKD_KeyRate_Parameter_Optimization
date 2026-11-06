# Preregistration: PQC surrogate for optimal decoy-state BB84 parameters

Branch `qml-surrogate`, created from `main` at `e796420` on 2026-11-06. This file is committed **before any
training**. Changes after that are made only as dated amendments at the end of this file. Nothing above an
amendment is edited.

This is a methods study. A negative or parity result is a valid outcome. **No quantum advantage is claimed, and the
design cannot support such a claim:** the circuits are simulated classically, and the comparison is against
classical surrogates of matched size.

## 0. Binding protocol

**Section 7 of `result_report/final_report.tex` is binding.** If anything below conflicts with it, Section 7 wins,
and the conflict is reported. Restated from Section 7:

- **Task.** Learn x ↦ p*(x), with x = (L, log₁₀ n_X) and preset `lim`. Models predict in u-space and map through
  `src/qkd_opt/reparam.py`, so every prediction is feasible.
- **Labels.** Labels come from the new optimizer and carry certificates. `cliff_edge` points are excluded from
  training and scoring, and listed. The two crossovers (n_X = 1e5, about 130.5 km; n_X = 1e6, about 118.5 km) are
  reported separately.
- **Splits.** No test point lies on the training grid.
  - Interpolation in L and in n_X.
  - The tail is scored separately.
  - Extrapolation is reported separately, never pooled with interpolation.
- **Primary metric.** Regret of the float64 R re-evaluated at the predicted parameters, bulk and tail separately:
  - median, p99, max;
  - % with r ≤ 1e-6, 1e-4, 1e-3, 1e-2;
  - % with no key (r ≥ 1).

  Secondary metrics: horizon error, and parameter error in u-space.
- **Baselines.**
  - The lookup table.
  - A classical NN with a matched parameter count.
  - The warm-started exact solver.
  - The hybrid "surrogate + k Newton steps", for k = 1, 2, 3.
- **Cost.**
  - Batch-1 latency, end-to-end and device-resident, plus batched throughput.
  - Training cost.
  - For QML: the backend, the shots per prediction, and the circuit evaluations per prediction.
- **Statistics.** At least 5 seeds; mean ± sd. Hyperparameters are chosen on a validation split only.
- **Usefulness.** A model is useful if either of the following holds:
  - (a) it reaches the warm solver's regret class at lower latency;
  - (b) it beats the lookup table's regret at comparable latency.

## 1. Hard rules

1. Every reported key rate is R re-evaluated by the float64 `lim` model at the predicted parameters, with the
   ε_sec = κℓ greatest fixed point (`src/qkd_opt/objective.evaluate`). No model-predicted R is ever reported.
2. Every prediction is a u-vector mapped through `reparam.params_of_u`, so it is feasible by construction
   (Section 4.6).
3. `.venv-qml` (PennyLane, JAX, Qiskit/IQM) imports no physics code. It exchanges data with the main `.venv` only
   through files:
   - labels and inputs as `.npz`, from `.venv`;
   - predicted u as `.npz`, from `.venv-qml`.
4. Nothing is submitted to IQM hardware without the owner's written "GO HARDWARE". The token is read only from the
   environment variable `IQM_TOKEN` and is never printed, logged, committed or written to a file.

## 2. Inputs (identical for all models)

- **v** = (log₁₀ n_X − 4) / 5. It lies in [0, 1] on the training range.
- **u_L** = L / L̂_h(n_X).
  - L̂_h is a monotone PCHIP interpolant in log₁₀ n_X through the six reference horizons of
    `Optimization/reference/reference_lim.json`: 130.441, 160.606, 172.624, 178.668, 182.091, 184.160 km.
  - Outside [4, 9] it extrapolates linearly, using the end-segment slope.
  - L̂_h uses only those six horizons. The horizons of held-out block sizes are not used.
- **Circuit and network inputs:** x′ = π(x − ½) for x ∈ {u_L, v}.

## 3. Labels (Phase 2; generated locally on AC, see Amendment 1)

### 3.1 Block sizes

- **TRAIN:** log₁₀ n_X ∈ {4.0, 4.5, …, 9.0} (11 values; includes the six reference block sizes).
- **HELDOUT** (interpolation in n_X): n_X ∈ {3·10⁶, 5·10⁸}.
- **EXTRAP:** n_X ∈ {5·10³, 2·10⁹}.

### 3.2 L grid

One `optimize_sweep` call covers all 15 chains (preset `lim`, default sweep settings) on a shared grid G:

- 0 ≤ L < 100 km in steps of 0.125 km;
- 100 ≤ L ≤ 200 km in steps of 0.03125 km;
- plus every L value of the reference grids. These are the 21 bulk L values of `reference_lim.json` and the L
  values of `reference_lim_tail.json`.

Every label stores:
- the parameters, θ, u, R and ℓ;
- status, `solver_iters`, `grad_norm_theta` and `hess_eig`;
- `branch_id` and `source`.

The sweep's `events` and per-n_X `horizon` (bisection) are stored with the labels.

### 3.3 Flags

- **cliff_edge:** R > 0 and `grad_norm_theta` > 1e-3.
- **noise_floor:** 1e-6 < `grad_norm_theta` ≤ 1e-3.

These are the rules of `build_reference.py`, applied to the label certificates.

### 3.4 Quality gate

At every point shared with the reference JSONs (the six reference block sizes, at reference L values):
- **Pass:** R_label ≥ R_ref (1 − tol_ref) at every point that is not cliff_edge.
- **On failure:** any failure stops the study and is reported.
- **Reporting:** the counts of better, equal and worse points are reported in any case.

### 3.5 Storage

`qml/data/labels_<git>_<cfg>.npz` plus `SHA256SUMS` and a `README`.

## 4. Splits (fixed by grid position, never by row; disjoint by construction)

Let h(n_X) be the label horizon (bisection). Then:
- **Tail region:** h − 15 km ≤ L ≤ h.
- **Bulk region:** L < h − 15 km.
- Only points with R_label > 0 that are not cliff_edge are trained on or scored.

Positions are integers on the 1/32 km lattice.

| split | TRAIN n_X, bulk region | TRAIN n_X, tail region |
|---|---|---|
| train | L ≡ 0 (mod 0.5 km) | L ≡ 0 (mod 0.125 km) |
| val | L ≡ 0.125 (mod 0.5) | L ≡ 0.03125 (mod 0.125) |
| test_bulk / test_tail | L ≡ 0.25 or 0.375 (mod 0.5) | L ≡ 0.0625 or 0.09375 (mod 0.125) |
| unused | other lattice points | — |

Further splits:
- **ref_bulk, ref_tail:** reference L values at the six reference block sizes. Not on the lattice, so never
  trained on. Scored against R_ref of the reference JSONs, with the reference cliff_edge flags.
- **heldout_<n_X>:** every lattice point of that n_X, scored as bulk and tail separately.
- **extrap_<n_X>:** the same for the two extrapolation block sizes. Reported separately.

Points with their own reporting:
- **crossover:** points within ±1 km of a below-horizon crossover, i.e. a `branch_switch` event with objective R in
  the label sweep, at the crossover km from the backfill. These are excluded from the bulk/tail aggregates and
  reported separately. Expected: n_X = 1e5 at about 130.5 km and n_X = 1e6 at about 118.5 km. Any further
  switches the label sweep finds are treated the same way and listed.
- **drift step:** n_X = 1e9, 77.7–78.2 km (STATUS open issue 1). Listed and flagged, not excluded.

## 5. Models

The tuning budget is equal for every family: **6 configurations × 1 seed (seed 0)**, each trained for **E = 2000
full-batch Adam steps**. The learning rate follows a cosine decay from 1e-2 to 1e-4. The loss is MSE in z-space,
averaged over the 5 outputs.

| model | configurations | trainable parameters |
|---|---|---|
| **PQC-full** | n_qubits × layers ∈ {5, 6, 8} × {3, 6} | 5·n·L |
| **PQC-product** | same grid, no entangling gates | 5·n·L |
| **NN** (JAX MLP, 2 → h → h → 5, tanh) | h ∈ {5, 6, 7, 8, 10, 12}, matched in order to the PQC-full configurations (75, 90, 120, 150, 180, 240) within ±10 % | h² + 9h + 5 |
| **Fourier/RFF ridge** | D/2 frequencies, D chosen so that 5(D+1) ≈ the PQC-full count of the same configuration; λ ∈ {1e-10, 1e-8, …, 1e-2} chosen on val inside the fit | 5(D+1) |
| **lookup-2D** | none | table size reported |
| **lookup-per-n_X** | none; reported only on splits whose n_X is in TRAIN, labelled as such | table size reported |
| **warm exact solver** | none | n/a |

**PQC circuit (PennyLane, `default.qubit`, JAX interface, analytic expectations).** Each layer ℓ applies, on every
qubit q:
- RX(a_{ℓq,1} · u_L′), then RY(a_{ℓq,2} · v′). The input scalings a are trainable and initialised at 1.
- Then a trainable Rot(φ, θ, ω).
- PQC-full only: a ring of CZ gates (q, q+1 mod n).

Outputs: z_i = ⟨Z_i⟩ for i = 0…4. One circuit gives all five, because the Z_i commute. So one circuit evaluation is
needed per prediction.

Note: the 8-qubit product circuit is five independent single-qubit models. Qubits 5–7 do not affect the outputs.
This is kept for budget symmetry and stated as such.

**RFF ridge.**
- **Features:** [1, cos(ω·x′), sin(ω·x′)].
- **Frequencies:** ω = Σ_{ℓ,q} c_{ℓq} (a_{ℓq,1}, a_{ℓq,2}), with c_{ℓq} drawn iid uniformly from {−1, 0, 1}. This
  samples the frequency spectrum of the trained PQC-full of the same configuration and seed.
- **Fit:** closed-form fit to z.
- **Output:** z is clipped to [−1, 1].

**Lookup tables**, built from the same training labels and the same inputs:
- **2D:** for each TRAIN row, linear interpolation in u_L over that row's training nodes, then linear interpolation
  in v between the two neighbouring rows. Outside a row's u_L range it holds the end value.
- **Per n_X:** linear interpolation in L over the training nodes of that n_X.

**Selection.** For each family, the configuration with the lowest **median validation regret** (bulk and tail
pooled; computed by `qml/certify.py`) is chosen. It is then trained with seeds 0–4. Each seed changes the
initialisation, and for the ridge also the frequency sampling. The data split is fixed. Two NN variants are
reported:
- **NN-tuned:** the best of the 6 NN configurations;
- **NN-matched:** the NN configuration whose index equals the selected PQC-full configuration.

**Kill criterion.** If the selected PQC-full has a median bulk test regret above 1e-2 (mean over seeds), it is
reported as a negative result, and there is no further retuning.

## 6. Output mapping (Addition B; identical for all models)

z ∈ [−1, 1] maps to u_i = lo_i + (hi_i − lo_i)(1 + z_i)/2.
- [lo_i, hi_i] is the range of u_i over the training labels, widened by 10 % of its width on each side and clipped
  to [U_LO, 1 − U_LO] (`reparam.U_LO` = 1e-10).
- The mapping is fixed and not trainable.
- Ridge and lookup outputs that leave [lo_i, hi_i] are clipped to it.
- u ↦ p goes through `reparam.params_of_u`, so every p is feasible.
- The lookup tables interpolate u directly, which stays inside the same interval.

## 7. Evaluation (`qml/certify.py`, main `.venv`)

**Per prediction:**
- R(p̂) and ℓ(p̂) from `objective.evaluate`;
- regret r = (R_ref − R(p̂)) / R_ref, where R_ref is R_label, or for the ref splits the reference JSON R;
- u-space error |û − u*|_∞.

**Per split:** median, p99, max; % r ≤ 1e-6, 1e-4, 1e-3, 1e-2; % no key (r ≥ 1).

**Horizon error, per n_X:** take the largest grid L with R(p̂(L)) > 0, scanning the grid up to h + 1 km, and
subtract h. Models are queried on every lattice point of each n_X up to h + 1 km.

**Hybrid (Addition A)**, from û at the test points of every split: `optimize.local_batch(θ̂, …, max_iter=k)` for
k = 1, 2, 3. The objective mode is 0 if R(p̂) > 0 and 1 otherwise; the scale is |value at θ̂|. Recorded: regret
after k steps, iterations used, and batch-1 wall time.

- **Scenario (i): no neighbouring solution.** Compare against a **batch-1 cold solve**
  (`optimize.optimize`, 2¹² Sobol, top-8, i.e. the benchmark's cold settings) at the same points. Its batch-1 time
  is measured first, in the Phase 3 pilot. It has never been measured before.
- **Scenario (ii): neighbour available.** Compare against the warm solver (`optimize.warm_solve`, max_iter 200)
  started from the label at the nearest lower training node. That is 0.5 km away in the bulk and 0.125 km in the
  tail. The reference figure is 1.47 ms end-to-end at 100 km, n_X = 1e9.
- **Usefulness** under Section 7 is judged separately for each scenario.
- **Hybrid wall time** = surrogate batch-1 latency (`.venv-qml`) + k-step batch-1 time (`.venv`). The cost of
  moving data between the two environments is not included, and this is stated.

**Latency, per model (`.venv-qml`):**
- batch-1 device-resident median and end-to-end median;
- batched throughput at batch 1024, reported as its own number;
- for the PQC: backend `default.qubit` (analytic, i.e. exact expectations), 0 shots on the simulator, 1 circuit
  per prediction;
- the shots S that hardware would need for a standard error ε in each ⟨Z_i⟩, reported as S ≈ (1 − z²)/ε².

**Also reported:** trainable parameter count and training wall time.

## 8. Expected outcome (stated in advance)

- The PQC is not expected to beat the lookup table on latency.
- In the bulk, the lookup table built from 0.5 km nodes is expected to have very small regret. Its 2 km version in
  the report had a median of 1.4e-9.
- A smooth surrogate is expected to compete, if anywhere, in the tail and at held-out n_X.
- The PQC is assessed on regret against parameter count, against the NN and the RFF ridge of matched size, and in
  the hybrid scenarios.
- Parity between PQC-full and its RFF ridge would indicate that the circuit's advantage over the product circuit
  (if any) comes from its frequency spectrum, not from entanglement.

## 9. Outputs and provenance

- Results go to `results/qml_<step>_<git>_<cfg>/`.
- Steps run through `qml/run_qml.py`, with records in `results/MANIFEST_qml.json` (power state at start).
- The power guard is `src/qkd_opt/provenance.power_state`, loaded read-only by file path, so `.venv-qml` imports no
  physics. Heavy steps refuse to run unless on AC.
- STATUS.md gets a dated section per step, with measured numbers only.
- Macros go to `result_report/generated_numbers_qml.tex`, via `qml/make_numbers_qml.py`.

## 10. Phase 5 (each part only with the owner's approval)

- **5a.** IQM fake backend first, then hardware only after "GO HARDWARE". Inference only.
  - First, a test that the Qiskit rebuild reproduces the PennyLane statevector expectations to ≤ 1e-10 at the saved
    weights (Addition C).
- **5b.** Extended inputs (e_mis, p_dc, α).


## 11. Environment (`.venv-qml`, Python 3.13.3; pinned in `qml/requirements-qml.txt`)

| package | version | role |
|---|---|---|
| pennylane | 0.45.1 | PQC training (`default.qubit`, JAX interface) |
| pennylane_lightning | 0.45.0 | installed alongside PennyLane; not used for training |
| jax / jaxlib | 0.10.2 | training, float64 |
| iqm-client[qiskit] | 36.0.0 | Phase 5a only |
| qiskit / qiskit-aer | 2.1.2 / 0.17.2 | forced by iqm-client (`qiskit<2.2`) |
| numpy / scipy / matplotlib | 2.5.4 / 1.18.1 / 3.11.2 | |

**JAX (Addition D).** The first install pulled jax 0.12.0.1; `jax.__version__` really reports that string. With it,
PennyLane 0.45.1 fails at runtime: `AttributeError: module 'jax.core' has no attribute 'is_concrete'`.

| jax | jit + grad + vmap smoke test with a 5-qubit circuit |
|---|---|
| 0.12.0.1, 0.11.2 | fail |
| 0.10.2, 0.9.2 | pass, float64 |

PennyLane's metadata declares no JAX pin. **jax 0.10.2** (newest passing) is pinned, and the version is recorded in
every provenance stamp. The main `.venv` keeps jax 0.9.0. The physics never runs in `.venv-qml`.

**IQM fake backends.** `iqm.qiskit_iqm` exports `IQMFakeAdonis`, `IQMFakeDeneb`, `IQMFakeApollo` and
`IQMFakeAphrodite`. Garnet is available only as the module `iqm.qiskit_iqm.fake_backends.fake_garnet`. This
matters only for Phase 5a.

---

## Amendment 1 (2026-11-06): labels are generated locally

Section 7 of the final report says labels are "generated where compute is cheap (e.g. cloud CPU/GPU), not on a
laptop". For this study, labels are generated **locally on AC power**.

**Reason.** The measured cost is small:
- the warm sweep takes 4.27 ms per point plus an 81 s one-time compile;
- a cold solve takes about 40 ms per point (both from `results/benchmark_lim_0c8f47a1f0_530f6d58460c/summary.json`).

The design in Section 3 (about 15 chains × ~4,200 grid points, warm sweep) is therefore minutes of compute.

Section 7's original sentence is unchanged. The same amendment is added, dated, under Section 7 of the report.

## Amendment 2 (2026-11-06): crossover windows from every below-horizon switch

Committed after the Phase 2 label run and before any model has been trained or has seen test data.

**Rule.** Section 4's crossover rule is made generic. It is now ±1 km around **every below-horizon switch logged by
the label sweep**: every `branch_switch` event with objective R whose crossover lies below the label horizon of its
n_X. The crossover km is the midpoint of the backfill bracket.

Points in these windows are excluded from the bulk/tail aggregates and reported separately.

**Switches.** The label sweep (`qml/data/labels_0d8bef1402_9fa16da52f39.json`) logged 13:

| n_X | crossover km (bracket) | detected at km | rel. gain at detection | u-distance |
|---|---|---|---|---|
| 3.162e+04 | 135.359 (135.344–135.375) | 135.531 | 7.06e-03 | 0.0954 |
| 1e+05 | 130.547 (130.531–130.562) | 130.594 | 6.19e-04 | 0.0616 |
| 3.162e+05 | 124.797 (124.781–124.812) | 125.000 | 1.14e-03 | 0.0412 |
| 1e+06 | 118.547 (118.531–118.562) | 118.812 | 8.48e-04 | 0.0284 |
| 3e+06 | 112.328 (112.312–112.344) | 112.469 | 3.11e-04 | 0.0203 |
| 3.162e+06 | 112.016 (112.000–112.031) | 112.156 | 2.85e-04 | 0.0200 |
| 1e+07 | 105.953 (105.938–105.969) | 106.250 | 1.16e-03 | 0.0143 |
| 1e+07 | 138.234 (138.219–138.250) | 138.344 | 8.51e-05 | 0.0079 |
| 1e+08 | 134.766 (134.750–134.781) | 134.906 | 6.11e-05 | 0.0076 |
| 3.162e+08 | 132.078 (132.062–132.094) | 132.094 | 6.71e-06 | 0.0064 |
| 5e+08 | 130.797 (130.781–130.812) | 130.875 | 1.83e-05 | 0.0059 |
| 1e+09 | 128.672 (128.656–128.688) | 128.822 | 2.76e-05 | 0.0050 |
| 2e+09 | 126.297 (126.281–126.312) | 126.469 | 2.27e-05 | 0.0041 |

"Rel. gain at detection" is the gain at the global check that found the switch. Global checks run every 10 grid
points, i.e. every 0.3125 km in this region. It is not the size of the discontinuity at the crossover. With these
windows, 401 scored points fall inside crossover windows.

## Amendment 3 (2026-11-06): sparse-training variant (label efficiency)

Also committed before any training.

- **Training set.** The same models and seeds, trained only on the training labels with L ≡ 0 (mod 2 km), i.e.
  lattice k ≡ 0 (mod 64), in both bulk and tail regions, per n_X.
- **No new tuning.** The configurations selected in the dense study are reused unchanged.
- **Built from the sparse set:** the output mapping (Section 6, same rule), both lookup tables, and the RFF
  frequencies (from the sparse-trained PQC-full of the same seed).
- **Evaluation.** The same test, ref, heldout and extrap sets. Reported as its own table; never pooled with the
  dense study.
- **Purpose.** Label efficiency, relevant to Phase 5b.
- **Condition.** It is included in the Phase 3 projection. If including it brings the projected total above 12 h,
  it is not run, and this is reported at STOP 2.

## Amendment 4 (2026-11-06): label repair after the run; re-certification without retraining (owner decision)

Committed while Phase 4 tuning was running on labels `0d8bef1402_9fa16da52f39`. No final model had been trained
yet. No test-set results had been seen.

**Finding.** STATUS 10d: 12 unlogged branch switches in the label sweep. Before 10 of them the sweep stayed on the
worse branch. 58 lattice labels are sub-optimal by up to 6.0e-4 relative R:
- train 4, val 6, test 11, heldout 11, extrap 2;
- the other 24 are unused.

**Decision.** The running Phase 4 (tune, select, final, latency, certify_final) finishes unchanged on the original
labels. Afterwards:

1. **Repair** (`qml/repair_labels.py`, main `.venv`, on AC):
   - Run a backward warm sweep (same grid, same n_X, `optimize_sweep` on L descending).
   - Per point, take the better of the forward label and the backward solution: R where either is positive,
     otherwise ℓ/n_X. This is the cont_up / cont_down rule of `build_reference.py`.
   - Splits, inputs and flags stay unchanged. Repaired points are flagged.
2. **Gate on the repaired labels.** Cold solves (2^14 Sobol, top-16) at:
   - every lattice point within ±3 km of every isolated jump of either sweep;
   - 1,000 random scored points.

   Pass if no repaired label is worse than a cold solve by more than 1e-9 relative. The reference-point gate of
   Section 3.4 is repeated.
3. **Storage.** Repaired labels go to `qml/data/repaired/`. No step picks them up automatically.
4. **Re-certification.** The predictions of the finished run are re-certified against the repaired labels
   (`certify.py --final --labels <repaired>`). RESULTS.md reports both certifications.
   - The primary metric uses the repaired labels.
   - No model is retrained. The 4 affected training targets stay as in the original labels, and this is stated.
5. **Crossover windows.** These are recomputed from the logged switches of both sweeps plus the unlogged jumps.
   Any change to the window list is reported.

## Amendment 5 (2026-11-06): final training moved to Leonardo GPUs; resume from checkpoints (owner decision)

The final training on the laptop CPU was stopped by the owner at 14:36. At about 25 min per seed, the remaining
~3 h on the CPU was not worth it. Nothing in the protocol changes: same code, data, selected configurations,
E = 2000 steps, seeds 0-4, float64.

**Kept from the stopped run.** Dense seed 0 for all five trained models (PQC-full 8×6, PQC-product 6×6, NN-tuned,
NN-matched, RFF-ridge).
- Their weights were saved when each finished.
- `train.py final --import-cpu-s0` turned them into parts. Predictions were recomputed from the weights, with no
  retraining. The recomputed training losses equal the run log.
- The RFF-ridge λ of seed 0 was not saved and is recorded as missing.

**Remaining 45 parts** (dense seeds 1-4, sparse seeds 0-4) are trained on CINECA Leonardo A100 GPUs, one
(variant, seed) per job, with `jax[cuda12]` 0.10.2 and float64.

**Reporting.** Training time is reported per hardware (CPU seed 0 vs GPU seeds) and never pooled. The hardware of
each part is stored in its metadata.

**Cross-hardware check.** If GPU time allows, dense seed 0 is also retrained on the GPU and compared with the CPU
part (weights, loss, predictions); the difference is reported. The CPU part remains the one used for seed 0.

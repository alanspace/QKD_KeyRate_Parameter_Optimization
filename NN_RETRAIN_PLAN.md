# NN surrogate: retrain plan (not yet executed)

## Why the old NN numbers are withdrawn

`results/nn_legacy_check/NN_LEGACY_CHECK.md`, produced by `Optimization/nn/check_legacy_nn.py` in `.venv-nn`, shows:
- `NeuralNetwork/models/bb84_nn_model.pth` has **6 outputs**, and every legacy evaluation script builds a **5-output** network.
- Three 5-output checkpoints exist, but nothing records which scaler pair each one was trained with.
- The two scaler pairs differ: `y_scaler` has 6 features in one pair and 5 in the other.
- The scalers were pickled with scikit-learn 1.7.2.
- The legacy evaluation scripts feed e_mis = 5e-3 to models trained on e_mis = 0.01. The labels came from the defective FoR run (REVIEW.md H4).

Therefore no NN number in the report can be regenerated. They are marked withdrawn (CLAIMS.md).

## Environments (kept separate)

| env | contents | role |
|---|---|---|
| `.venv` | JAX x64, SciPy, NumPy | Generate labels, evaluate regret and run the lookup-table baseline. **No torch.** |
| `.venv-nn` | `requirements-nn.txt`: torch 2.15.0, scikit-learn 1.9.1, joblib, numpy, scipy, matplotlib | Training and NN inference only. |

The two envs exchange data only through files:
- `.venv` writes labels as `.npz`.
- `.venv-nn` writes predictions as `.npz`.
- `.venv` evaluates those predictions with the float64 physics.

No process imports both stacks.

## Inputs (≥ 4 dimensions) and targets

**Inputs** (preset `lim`):

| input | range |
|---|---|
| L | [0, 200] km |
| log₁₀ n_X | [4, 9], continuous |
| log₁₀ p_dc | [−7, −5] |
| e_mis | [2.5e-3, 3e-2] |

An optional 5th input is α ∈ [0.16, 0.24] dB/km. η_Bob and f_EC stay fixed.

**Targets:**
- The logit-space parameters θ (`src/qkd_opt/reparam.py`), so every prediction is feasible by construction. No clipping or renormalisation afterwards.
- A separate head for the horizon L*(n_X, p_dc, e_mis) and for sign(l). Beyond the horizon the parameters are degenerate (REVIEW §2c), so they are **not** regression targets.

## Labels

- **Design:** scrambled Sobol over (log n_X, log p_dc, e_mis), 512 design points. For each one, run `optimize_sweep` on an L grid with 1 km spacing up to the horizon.
  - This gives about 1e5 labelled points.
  - Every label keeps its status, certificate and `branch_id`.
- **Quality gate:** at a random 2 % of label points, run a dense cold solve (2^17 Sobol) and the SLSQP cross-check. Any label regret above 1e-6 fails the batch.
- **Cost:** the measured steady-state cost of a `lim` warm sweep is `\QKDLimSweepMsPerPoint` ms/point (`results/NUMBERS.md`). A per-design compile is not needed, because the constants are traced. This cost is an estimate from that measurement and is not claimed anywhere.

## Splits (by design point, never by row)

- **Interior L:** a test grid offset by 0.5 km from the training grid.
- **Held-out n_X band:** log₁₀ n_X ∈ [6.25, 6.75], removed from training (interpolation in n_X).
- **Extrapolation corners:** p_dc > 3e-6 or e_mis > 0.025 held out, and reported separately as extrapolation.
- **Seeds:** fixed for Sobol, the split, torch and the dataloader. Training may use MPS; the final evaluation runs on CPU, then regret is computed in JAX float64.

## Baselines (same inputs, same label budget)

1. **Lookup table:** multilinear interpolation of θ on a regular 4-D grid (L every 1 km, log n_X every 0.5, 4 log p_dc values, 6 e_mis values), with the label budget matched to the NN's.
2. Nearest neighbour in the scaled input space.
3. The warm-started JAX solver as the accuracy ceiling. It has zero regret by definition, and its latency comes from `timing_benchmark.py`.

## Metrics (all produced by a script; nothing typed)

- **Regret** vs the re-optimised optimum at every test point: median, p99, max, % within 1e-3 and within 1e-6.
  - Reported separately near branch switches (± 2 km of an event) and in the tail window.
- **Horizon error** in km, plus the % of predictions with R ≤ 0 where R* > 0.
- **Latency:**
  - batch-1 steady state (`torch.no_grad`, warm-up, device-resident, CPU and MPS) vs batch-1 end-to-end;
  - batched throughput, reported as its own number and never divided against batch-1 times;
  - the same protocol for the lookup table and for the JAX solver.
- **Acceptance:** the NN is worth keeping only if it beats the lookup table on regret at comparable latency and memory. If it does not, the report says so.

## Steps

1. `Optimization/nn/make_labels.py` (`.venv`) → `results/nn/labels_<commit>_<cfg>.npz`
2. `Optimization/nn/train.py` (`.venv-nn`) → model, scalers and metadata in one versioned directory, with the architecture stored *inside* the checkpoint.
3. `Optimization/nn/predict.py` (`.venv-nn`) → predictions `.npz`
4. `Optimization/nn/evaluate.py` (`.venv`) → `results/nn/NN_RESULTS.md` with regret, horizon error and latency against the baselines.
5. Add the generated values to `make_numbers.py`. The report imports them only after step 4 passes review.

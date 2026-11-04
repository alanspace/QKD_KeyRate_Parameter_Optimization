# Archived legacy optimizer variants (Feb 2026)

Frozen, unmodified copies of the pre-rework optimizers and comparison plotting.
Kept for provenance only — **do not use for new results**. See `REVIEW.md` at the repo
root for the defects found in these scripts (notably: `BB84_PureDualAnnealing.py` never
records results on its success path; `plotting/generate_paper_plots.py` interpolates over
Hessian failures, substitutes DA data for the n_X=1e4 "Hessian" curve and appends synthetic
"dive" points).

Path note: these scripts resolve the project root relative to their *original* location
(`Optimization/run_scripts/JAX_variants/`); run them from a checkout of commit
before the move if you need to reproduce legacy outputs.

The legacy result JSONs remain untracked (gitignored) under
`Optimization/run_scripts/outputs/JAX/`; their SHA-256 checksums are recorded in
`Optimization/reference/legacy_da/PROVENANCE.md`.

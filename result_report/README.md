# Reports

## Current report

**`final_report.tex` / `final_report.pdf`** (November 2026) is the report to cite. It supersedes every
document listed below.

- Every `\QKD...` number in it is imported from `generated_numbers.tex`.
- That file is written by `Optimization/report/make_numbers.py` from `results/MANIFEST.json`.
- Numbers marked † are copied from the result file named next to them.

Build:

```
cd result_report && pdflatex final_report.tex && pdflatex final_report.tex
```

The figures are read from `../results/figures/900d9f88a8_58c9f38705cd/`.

## Superseded documents (kept as historical record; do not cite their numbers)

| File | Date | Status |
|---|---|---|
| `results_report.tex` (source) | Jan/Mar 2026 | Claims withdrawn; see `../CLAIMS.md`. Left in place because `Optimization/report/apply_report_revision.py` targets this path. |
| `superseded/results_report_2026-03.pdf` | Mar 2026 | Compiled PDF of `results_report.tex`. Claims withdrawn. |
| `superseded/PhD_interview_presentation_2026-03.pdf` | Mar 2026 | Slides that repeat the results-report numbers (3,574×, 129×, 20 km+, 5.9 M×). Claims withdrawn. |
| `../Project_Report/*Final_Report.pdf`, `*Presentation.pdf` | May 2025 | MSc thesis and its defence slides. Superseded and not re-scored; see Appendix A of the final report. |

All PDFs are git-ignored (`*.pdf` in `.gitignore`). Only the `.tex` sources are versioned.

# Slides: QKD parameter optimisation and the QML surrogate study

- Open `slides.html` in a browser.
  - Keys: right / space = next, left = back, `o` = overview.
  - Print to PDF in landscape for a one-page-per-slide PDF.
- To edit, change the `<section class="slide">` blocks in `slides.html`.
  - Every number on a slide names its source file in the slide's footer.
  - Dashed "pending" boxes are filled from `qml/RESULTS.md` once Phase 4 finishes.
- To rebuild the assets:
  - `.venv/bin/python qml/slides/make_assets.py` copies the report figures and draws `dataset_splits.png`.
  - `.venv-qml/bin/python qml/slides/draw_circuit.py` draws the circuit.
  - `assets/SOURCES.md` lists each file's source path and SHA-256.

## Two decks
- `pitch.html` (10 slides, hackathon format): data, challenge, argue for quantum, prototype, close the loop, plus a KPI
  scoreboard and the parameters-across-distance showcase.
- `slides.html` (30 slides): the detailed version with methods, checks and sources.
- `make_showcase.py` draws `params_across_distance.png` now. Once Phase 4 results exist it also draws
  `showcase_predictions.png` (predicted vs optimal settings across distance).

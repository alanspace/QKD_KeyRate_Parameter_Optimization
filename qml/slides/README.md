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

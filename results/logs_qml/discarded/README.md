# Discarded attempts (not data)

- `retime_20261109_195525.log`, `retime_20261109_195756.log`, `retime_20261109_200026.log`: `qml/retime.py --allow-battery`
  on 2026-11-09, on battery with macOS Low Power Mode on. The owner withdrew the battery permission and asked for the
  re-timing to be done on AC with Low Power Mode off. The first two attempts refused (1-min load 2.03 and 2.79 > 2.0
  right after the test suite); the third was stopped during its 60 s settle. No timing was measured and no
  `results/qml_retime_*` output was written. Nothing here feeds into RESULTS.md.

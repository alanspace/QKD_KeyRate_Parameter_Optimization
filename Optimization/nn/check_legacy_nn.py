"""Document why the legacy NN numbers cannot be reproduced. Runs in the SEPARATE NN env:
    .venv-nn/bin/python Optimization/nn/check_legacy_nn.py
Loads every checkpoint and scaler (read-only), records layer shapes / feature counts, and tries the
5-output architecture used by all legacy evaluation scripts (strict state_dict load).
Writes results/nn_legacy_check/NN_LEGACY_CHECK.md + .json.
"""
import json
import os
import subprocess
import warnings

import joblib
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPTS = ["NeuralNetwork/models/bb84_nn_model.pth", "NeuralNetwork/models/bb84_nn_model_jax.pth",
         "NeuralNetwork/models/bb84_nn_model_legacy.pth", "models/bb84_nn_model.pth"]
SCALERS = ["NeuralNetwork/models/scaler.pkl", "NeuralNetwork/models/y_scaler.pkl", "models/scaler.pkl",
           "models/y_scaler.pkl"]


class BB84Network5(nn.Module):  # architecture of verify_report_metrics.py:59-72 et al.
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2, self.fc3, self.fc4 = nn.Linear(4, 16), nn.Linear(16, 32), nn.Linear(32, 16), nn.Linear(16, 5)


def main():
    out = dict(torch=torch.__version__, checkpoints={}, scalers={})
    commit = subprocess.run(["git", "rev-parse", "--short=10", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    for c in CKPTS:
        p = os.path.join(ROOT, c)
        if not os.path.exists(p):
            out["checkpoints"][c] = "missing"
            continue
        sd = torch.load(p, map_location="cpu", weights_only=True)
        shapes = {k: list(v.shape) for k, v in sd.items()}
        try:
            BB84Network5().load_state_dict(sd, strict=True)
            load5 = "ok"
        except Exception as e:  # noqa: BLE001
            load5 = f"FAILS: {str(e).splitlines()[0][:160]}"
        out["checkpoints"][c] = dict(shapes=shapes, n_outputs=shapes.get("fc4.weight", [None])[0], load_as_5_output=load5)
    for s in SCALERS:
        p = os.path.join(ROOT, s)
        if not os.path.exists(p):
            out["scalers"][s] = "missing"
            continue
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = joblib.load(p)
        out["scalers"][s] = dict(warnings=sorted({str(x.message).split(". ")[0] for x in w}), type=type(obj).__name__, n_features_in=int(getattr(obj, "n_features_in_", -1)),
                                 n_samples_seen=int(getattr(obj, "n_samples_seen_", -1))
                                 if not hasattr(getattr(obj, "n_samples_seen_", None), "__len__")
                                 else [int(x) for x in obj.n_samples_seen_][:1])
    d = os.path.join(ROOT, "results", "nn_legacy_check")
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, "nn_legacy_check.json"), "w"), indent=1)
    md = [f"# Legacy NN loadability check (git `{commit}`, torch {torch.__version__}, separate env `.venv-nn`)", "",
          "| checkpoint | outputs (fc4 rows) | loads into the 5-output net used by all legacy evaluation scripts |",
          "|---|---|---|"]
    for c, v in out["checkpoints"].items():
        md.append(f"| `{c}` | {v if isinstance(v, str) else v['n_outputs']} | "
                  f"{'' if isinstance(v, str) else v['load_as_5_output']} |")
    md += ["", "| scaler | type | n_features_in | n_samples_seen |", "|---|---|---|---|"]
    for s, v in out["scalers"].items():
        md.append(f"| `{s}` | {v if isinstance(v, str) else v['type']} | "
                  f"{'' if isinstance(v, str) else v['n_features_in']} | {'' if isinstance(v, str) else v['n_samples_seen']} |")
    warns = sorted({m for v in out["scalers"].values() if not isinstance(v, str) for m in v["warnings"]})
    md += ["", "Unpickling warnings: " + ("; ".join(warns) if warns else "none"), "",
           "No file records which scaler pair belongs to which checkpoint; see NN_RETRAIN_PLAN.md."]
    open(os.path.join(d, "NN_LEGACY_CHECK.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

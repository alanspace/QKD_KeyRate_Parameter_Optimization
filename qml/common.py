"""Shared, physics-free helpers for the qml study (importable from both .venv and .venv-qml).

Only the standard library and numpy are imported here. The power guard and hashing come from
src/qkd_opt/provenance.py, loaded *by file path* so that src/qkd_opt/__init__.py (which imports the physics) never
runs inside .venv-qml.
"""
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QML = os.path.join(ROOT, "qml")
DATA = os.path.join(QML, "data")
RESULTS = os.path.join(ROOT, "results")

# ---- design (PREREGISTRATION.md sections 2-4) ----------------------------------------------------
REF_HORIZONS_LOG10NX = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]          # reference_lim.json, read at run time
TRAIN_LOG10NX = [4.0 + 0.5 * i for i in range(11)]
HELDOUT_NX = [3e6, 5e8]
EXTRAP_NX = [5e3, 2e9]
LATTICE = 32                      # positions are integers on the 1/32 km lattice
BULK_STEP_LOW = 0.125             # grid step for L < 100 km
FINE_START = 100.0                # grid step 1/32 km for L >= 100 km
TAIL_WINDOW_KM = 15.0
CROSSOVER_HALF_KM = 1.0
DRIFT_WINDOW = dict(n_X=1e9, L_lo=77.7, L_hi=78.2)
CLIFF_GTOL, NOISE_GTOL = 1e-3, 1e-6   # build_reference.py rules (checked against it in tests)
U_LO = 1e-10                      # reparam.U_LO (checked against it in tests)
U_MARGIN = 0.10                   # output-range widening, section 6


def load_provenance():
    """src/qkd_opt/provenance.py loaded by path (stdlib only at import; no physics)."""
    path = os.path.join(ROOT, "src", "qkd_opt", "provenance.py")
    spec = importlib.util.spec_from_file_location("qkd_provenance_by_path", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def power_state():
    return load_provenance().power_state()


def require_ac(step: str):
    pw = power_state()
    if pw["source"] != "AC":
        sys.exit(f"{step}: refusing to run, power source is {pw['source']!r} "
                 f"({pw['battery_percent']}% {pw['battery_state']}); connect AC power")
    return pw


def config_hash(cfg: dict) -> str:
    return load_provenance().config_hash(cfg)


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def stamp_qml(cfg: dict) -> dict:
    """Provenance stamp for .venv-qml scripts (no physics import)."""
    from importlib.metadata import version, PackageNotFoundError

    def v(p):
        try:
            return version(p)
        except PackageNotFoundError:
            return None
    prov = load_provenance()
    return dict(
        git_commit=git("rev-parse", "--short=10", "HEAD"), git_branch=git("rev-parse", "--abbrev-ref", "HEAD"),
        qml_uncommitted_changes=bool(git("status", "--porcelain", "--", "qml")),
        config_hash=config_hash(cfg), python=platform.python_version(), env=sys.prefix,
        packages={p: v(p) for p in ("jax", "jaxlib", "pennylane", "pennylane_lightning", "numpy", "scipy",
                                    "qiskit", "qiskit-aer", "iqm-client")},
        cpu=prov.cpu_model(), machine=platform.machine(), os=platform.platform(), n_cpu=os.cpu_count(),
        power=prov.power_state())


def out_dir(step: str, cfg: dict) -> str:
    d = os.path.join(RESULTS, f"qml_{step}_{git('rev-parse', '--short=10', 'HEAD')}_{config_hash(cfg)}")
    os.makedirs(d, exist_ok=True)
    return d


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]


def write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))


# ---- inputs (section 2) ------------------------------------------------------------------------------
def v_of_nx(n_X):
    import numpy as np
    return (np.log10(np.asarray(n_X, np.float64)) - 4.0) / 5.0


def horizon_fit(ref_log10nx, ref_h):
    """Monotone PCHIP in log10 n_X through the reference horizons; linear extrapolation with end slopes."""
    import numpy as np
    from scipy.interpolate import PchipInterpolator
    x = np.asarray(ref_log10nx, np.float64)
    y = np.asarray(ref_h, np.float64)
    f = PchipInterpolator(x, y, extrapolate=False)
    d = f.derivative()
    s_lo, s_hi = float(d(x[0])), float(d(x[-1]))

    def Lh(log10nx):
        t = np.asarray(log10nx, np.float64)
        out = f(np.clip(t, x[0], x[-1]))
        out = np.where(t < x[0], y[0] + s_lo * (t - x[0]), out)
        return np.where(t > x[-1], y[-1] + s_hi * (t - x[-1]), out)
    return Lh


def z_to_u(z, lo, hi):
    """Section 6: fixed affine map z in [-1,1] -> u in [lo, hi] (per output)."""
    import numpy as np
    z = np.clip(np.asarray(z, np.float64), -1.0, 1.0)
    return lo + (hi - lo) * (1.0 + z) / 2.0


def u_to_z(u, lo, hi):
    import numpy as np
    return 2.0 * (np.asarray(u, np.float64) - lo) / (hi - lo) - 1.0


def u_range(u_train):
    """[lo, hi] per output: training range widened by U_MARGIN of its width, clipped to [U_LO, 1-U_LO]."""
    import numpy as np
    lo, hi = u_train.min(0), u_train.max(0)
    w = hi - lo
    return np.maximum(lo - U_MARGIN * w, U_LO), np.minimum(hi + U_MARGIN * w, 1.0 - U_LO)

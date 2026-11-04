"""Provenance stamps for results and figures: git commit, dirty state, config hash, hardware."""
import hashlib
import json
import os
import platform
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def _hash_files(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        with open(p, "rb") as fh:
            h.update(os.path.relpath(p, ROOT).encode())
            h.update(fh.read())
    return h.hexdigest()[:12]


def physics_source_hash():
    """Hash of the physics source actually imported (src/qkd + src/qkd_opt), incl. uncommitted edits."""
    files = []
    for sub in ("src/qkd", "src/qkd_opt"):
        d = os.path.join(ROOT, sub)
        files += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".py")]
    return _hash_files(files)


def config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def cpu_model():
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True).stdout.strip()
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def stamp(config: dict | None = None) -> dict:
    import jax
    import numpy as np
    import scipy

    dirty_src = _git("status", "--porcelain", "--", "src")
    devs = jax.devices()
    return dict(
        git_commit=_git("rev-parse", "--short=10", "HEAD"),
        git_branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
        src_uncommitted_changes=bool(dirty_src),
        physics_source_hash=physics_source_hash(),
        config_hash=config_hash(config or {}),
        jax=jax.__version__, numpy=np.__version__, scipy=scipy.__version__, python=platform.python_version(),
        jax_backend=jax.default_backend(), jax_devices=[str(d) for d in devs],
        x64=bool(jax.config.read("jax_enable_x64")),
        cpu=cpu_model(), machine=platform.machine(), os=platform.platform(), n_cpu=os.cpu_count(),
    )


def stamp_text(st: dict) -> str:
    dirty = "+dirty-src" if st.get("src_uncommitted_changes") else ""
    return (f"git {st['git_commit']}{dirty} | cfg {st['config_hash']} | phys {st['physics_source_hash']} | "
            f"{st['jax_backend']} {st['cpu']} | jax {st['jax']}")


def stamp_figure(fig, st: dict):
    fig.text(0.995, 0.003, stamp_text(st), ha="right", va="bottom", fontsize=6, color="0.4")


def to_jsonable(x):
    import numpy as np
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return str(x)
    return x

"""Phase 5a preparation (.venv-qml): rebuild the trained PQC in Qiskit, check it, plan and simulate IQM runs.

  --check     Addition C: Qiskit statevector <Z_i> vs PennyLane (qml/models.py) at the given weights; must be <= 1e-10.
  --plan      transpile one circuit to IQM fake backends (native gates, connectivity): gate counts, depth, layout.
  --fake N S  run N test points on an IQM fake backend (noisy local simulation) with S shots each; report the error
              of <Z_i> vs the exact values and the shot-noise floor. Writes predictions for certification.
  --hardware  REFUSES unless --go-hardware is given (owner's written "GO HARDWARE" required, PREREGISTRATION hard
              rule 4). The token is read only from the environment variable IQM_TOKEN; it is never printed or written.

Gate conventions: PennyLane Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi); RX/RY identical in both; Qiskit
qubit i <-> PennyLane wire i; <Z_i> from Pauli strings in Qiskit's little-endian order.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def xprime(uL, v):
    return np.pi * (np.array([uL, v], dtype=np.float64) - 0.5)


def build(a, r, uL, v, entangle=True, measure=False):
    from qiskit import QuantumCircuit
    layers, n, _ = a.shape
    x = xprime(uL, v)
    qc = QuantumCircuit(n, 5 if measure else 0)
    for l_ in range(layers):
        for q in range(n):
            qc.rx(float(a[l_, q, 0] * x[0]), q)
            qc.ry(float(a[l_, q, 1] * x[1]), q)
            phi, theta, omega = (float(t) for t in r[l_, q])
            qc.rz(phi, q); qc.ry(theta, q); qc.rz(omega, q)
        if entangle and n > 1:
            for q in range(n):
                qc.cz(q, (q + 1) % n)
    if measure:
        qc.measure(range(5), range(5))
    return qc


def z_exact(qc):
    from qiskit.quantum_info import Statevector, SparsePauliOp
    sv = Statevector(qc)
    n = qc.num_qubits
    out = []
    for i in range(5):
        lab = ["I"] * n
        lab[n - 1 - i] = "Z"
        out.append(float(np.real(sv.expectation_value(SparsePauliOp("".join(lab))))))
    return np.array(out)


def z_from_counts(counts, shots):
    z = np.zeros(5)
    for bits, c in counts.items():
        b = bits.replace(" ", "")[::-1]          # little-endian: bit i of the key string is clbit i
        for i in range(5):
            z[i] += c * (1.0 if b[i] == "0" else -1.0)
    return z / shots


def pennylane_z(a, r, X, n, layers, entangle):
    import jax.numpy as jnp
    from qml import models as M
    _, apply, _ = M.make_pqc(n, layers, entangle)
    w = dict(a=jnp.asarray(a), r=jnp.asarray(r))
    return np.asarray(apply(w, M.xprime(jnp.asarray(X[:, 0]), jnp.asarray(X[:, 1]))))


def load_weights(name):
    f = os.path.join(ROOT, "qml", "data", f"weights_{name}_dense_s0.npz")
    if not os.path.exists(f):
        return None
    w = np.load(f)
    return np.asarray(w["a"]), np.asarray(w["r"])


def cmd_check(a_):
    rng = np.random.default_rng(0)
    rows = []
    cases = []
    for name, ent in (("PQC-full", True), ("PQC-product", False)):
        w = load_weights(name)
        if w is not None:
            cases.append((f"trained {name}", w[0], w[1], ent))
    if not cases or a_.random:
        for (n, L) in ((5, 3), (8, 6)):
            for ent in (True, False):
                cases.append((f"random {n}x{L} {'full' if ent else 'product'}", rng.normal(1, .5, (L, n, 2)),
                              rng.uniform(-np.pi, np.pi, (L, n, 3)), ent))
    X = np.column_stack([rng.uniform(0, 1.05, 20), rng.uniform(-0.06, 1.06, 20)])
    worst = 0.0
    for label, a, r, ent in cases:
        L, n, _ = a.shape
        zp = pennylane_z(a, r, X, n, L, ent)
        zq = np.array([z_exact(build(a, r, x[0], x[1], ent)) for x in X])
        d = float(np.max(np.abs(zp - zq)))
        worst = max(worst, d)
        rows.append(dict(case=label, qubits=n, layers=L, max_abs_diff=d))
        print(f"[check] {label}: max |<Z>_pennylane - <Z>_qiskit| = {d:.2e} over {len(X)} inputs", flush=True)
    ok = worst <= 1e-10
    print(f"[check] Addition C: {'PASS' if ok else 'FAIL'} (worst {worst:.2e}, limit 1e-10)", flush=True)
    return ok, rows


def fake_backends():
    from iqm.qiskit_iqm import IQMFakeAdonis, IQMFakeDeneb, IQMFakeApollo
    out = {"IQMFakeAdonis": IQMFakeAdonis, "IQMFakeDeneb": IQMFakeDeneb, "IQMFakeApollo": IQMFakeApollo}
    try:
        from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
        out["IQMFakeGarnet"] = IQMFakeGarnet
    except Exception as e:  # noqa: BLE001
        print(f"[plan] Garnet fake backend not importable: {e}")
    return out


def cmd_plan(a_):
    from qiskit import transpile
    w = load_weights("PQC-full")
    if w is None:
        rng = np.random.default_rng(0)
        L, n = a_.layers, a_.qubits
        w = (rng.normal(1, .5, (L, n, 2)), rng.uniform(-np.pi, np.pi, (L, n, 3)))
        print(f"[plan] no trained weights yet: structure of a {n}x{L} circuit with random angles")
    qc = build(w[0], w[1], 0.6, 0.5, True, measure=True)
    res = {}
    for name, cls in fake_backends().items():
        try:
            be = cls()
            if be.num_qubits < qc.num_qubits:
                res[name] = dict(note=f"too few qubits ({be.num_qubits})")
                continue
            t = transpile(qc, backend=be, optimization_level=3, seed_transpiler=0)
            ops = dict(t.count_ops())
            res[name] = dict(qubits=be.num_qubits, depth=t.depth(), ops={k: int(v) for k, v in ops.items()},
                             two_qubit=int(sum(v for k, v in ops.items() if k in ("cz", "move"))))
            print(f"[plan] {name}: {res[name]}", flush=True)
        except Exception as e:  # noqa: BLE001
            res[name] = dict(error=str(e)[:200])
            print(f"[plan] {name}: {e}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--random", action="store_true", help="also check random weights")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--qubits", type=int, default=5)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--hardware", action="store_true")
    ap.add_argument("--go-hardware", action="store_true")
    a = ap.parse_args()
    if a.hardware:
        if not a.go_hardware:
            sys.exit("hardware submission refused: requires the owner's written GO HARDWARE and --go-hardware")
        if not os.environ.get("IQM_TOKEN"):
            sys.exit("IQM_TOKEN not set in the environment")
        sys.exit("hardware path not implemented until the plan (backend, circuits, shots, credits) is approved")
    out = {}
    if a.check:
        out["check_ok"], out["check"] = cmd_check(a)
    if a.plan:
        out["plan"] = cmd_plan(a)
    print(json.dumps({k: v for k, v in out.items() if k != "check"}, indent=1, default=str))


if __name__ == "__main__":
    main()

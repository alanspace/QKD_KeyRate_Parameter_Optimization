"""Draw the PQC used in the study (.venv-qml): one data re-uploading layer on 5 qubits, as built by qml/models.py.

Writes qml/slides/assets/pqc_circuit.png (qml.draw_mpl) and pqc_circuit.txt (qml.draw, 2 layers).
Weights are placeholders (a = 1, rotation angles 0) - the drawing shows structure, not trained values.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import jax.numpy as jnp  # noqa: E402
import pennylane as qml  # noqa: E402

A = os.path.join(ROOT, "qml", "slides", "assets")


def build(n, layers, entangle=True):
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit(uL, v):
        for l_ in range(layers):
            for q in range(n):
                qml.RX(1.0 * uL, wires=q, id=f"a{l_}{q}·uL")
                qml.RY(1.0 * v, wires=q, id=f"b{l_}{q}·v")
                qml.Rot(0.0, 0.0, 0.0, wires=q, id=f"θ{l_}{q}")
            if entangle:
                for q in range(n):
                    qml.CZ(wires=[q, (q + 1) % n])
        return [qml.expval(qml.PauliZ(i)) for i in range(5)]
    return circuit


def main():
    os.makedirs(A, exist_ok=True)
    c1 = build(5, 1)
    fig, _ = qml.draw_mpl(c1, decimals=None, style="pennylane")(0.3, 0.6)
    fig.suptitle("One layer of PQC-full (5 qubits): encode u_L and v with trainable scalings, trainable Rot, CZ ring; "
                 "repeated L times; measure <Z_0..Z_4>", fontsize=9)
    fig.savefig(os.path.join(A, "pqc_circuit.png"), dpi=170, bbox_inches="tight")
    txt = qml.draw(build(5, 2), decimals=None)(0.3, 0.6)
    with open(os.path.join(A, "pqc_circuit.txt"), "w") as fh:
        fh.write("PQC-full, 5 qubits, 2 layers (structure only; qml.draw)\n\n" + txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()

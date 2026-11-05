"""Physics presets. Every result produced by qkd_opt is labelled with one preset name.

* ``lim`` (default): Lim, Curty, Walenta, Xu, Zbinden, PRA 89, 022307 (2014), arXiv:1311.7129,
  Eqs. (1)-(5) with the "dedicated fiber" error model, EXCEPT that the misalignment term of e_k
  uses eta_sys = eta_Bob * eta_ch instead of the printed eta_ch:
      e_k = p_dc + e_mis [1 - exp(-eta_sys k)] + p_ap D_k / 2
      D_k = 1 - (1 - 2 p_dc) exp(-eta_sys k),  R_k = D_k (1 + p_ap)
      eps_sec = kappa * l (greatest fixed point), kappa = 1e-15
      m_Z = sum_k m_{Z,k} (total Z-basis errors, Eq. 4)
  With eta_sys the model reproduces the paper's stated numbers to within ~10 % -- NOT exactly
  (see PHYSICS_DIFF.md: R(0, 1e9), the 100 km ratio, and the n_X = 1e4 reach). The printed eta_ch
  misses all three by far, so eta_ch is treated as a probable typo. Model choices the paper does
  not print are listed in PHYSICS_DIFF.md ("Unprinted model choices").
* ``lim_literal`` (diagnostic only): identical to ``lim`` but with the printed eta_ch.
* ``legacy``: the physics of src/qkd (e_mis = 0.01, p_ap = 0, eps_sec = 1e-10 fixed,
  m_Z = sum of n_X terms, eta_sys in e_k, n^-/m^- clamped at 0), evaluated by
  src/qkd_opt/physics_lim.py with legacy switches. This is tested equal to src/qkd to < 1e-8
  relative in l at interior points (tests/test_qkd_opt_physics.py), and makes benchmarks
  independent of the src/qkd working tree. src/qkd's extra internal clamps (gamma <= 10,
  v/s clipped to [-10, 10], b sanitised) only act in degenerate regions and are not reproduced.
* ``legacy_src`` (cross-check only): calls src/qkd/model.py itself.

``Flags`` are static (they select code paths; changing them recompiles); ``Consts`` are traced
floats (changing them does not recompile).
"""
from typing import NamedTuple


class Consts(NamedTuple):
    alpha: float = 0.2        # dB/km
    eta_bob: float = 0.1
    p_dc: float = 6e-7
    e_mis: float = 5e-3
    p_ap: float = 4e-2
    f_ec: float = 1.16
    eps_cor: float = 1e-15
    mu3: float = 2e-4
    eps_sec: float = 1e-10    # used only when Flags.eps_mode == "fixed"
    kappa: float = 1e-15      # used only when Flags.eps_mode == "kappa"


class Flags(NamedTuple):
    model: str = "lim"            # "lim" (src/qkd_opt/physics_lim.py) | "legacy_src" (src/qkd itself)
    emis_eta: str = "sys"         # transmittance in the e_mis term of e_k: "sys" | "ch" (as printed)
    mz_mode: str = "errors"       # m_Z in Eq. (4): "errors" = sum_k m_{Z,k} | "nx_legacy" = sum_k n_{X,k}
    eps_mode: str = "kappa"       # "kappa": eps_sec = kappa * l (fixed point) | "fixed": eps_sec = Consts.eps_sec
    afterpulse_in_rate: bool = True   # R_k = D_k (1 + p_ap) used for counts and N
    clamp_minus: bool = False     # clamp n^-, m^- at 0 (legacy behaviour); Lim's formulas do not
    n_eps_iter: int = 8           # monotone fixed-point iterations before log-bisection (eps_sec = kappa * l)


class Preset(NamedTuple):
    name: str
    flags: Flags
    consts: Consts


LIM = Preset(name="lim", flags=Flags(), consts=Consts())

LIM_LITERAL = Preset(name="lim_literal", flags=Flags(emis_eta="ch"), consts=Consts())

_LEGACY_FLAGS = Flags(model="lim", emis_eta="sys", mz_mode="nx_legacy", eps_mode="fixed",
                      afterpulse_in_rate=False, clamp_minus=True)
_LEGACY_CONSTS = Consts(e_mis=0.01, p_ap=0.0, eps_sec=1e-10)

LEGACY = Preset(name="legacy", flags=_LEGACY_FLAGS, consts=_LEGACY_CONSTS)

LEGACY_SRC = Preset(name="legacy_src", flags=_LEGACY_FLAGS._replace(model="legacy_src"), consts=_LEGACY_CONSTS)

PRESETS = {p.name: p for p in (LIM, LIM_LITERAL, LEGACY, LEGACY_SRC)}
RENAMED = {"lim_exact": "lim_literal"}


def get_preset(name: str) -> Preset:
    if name in RENAMED:
        raise ValueError(f"preset {name!r} was renamed to {RENAMED[name]!r} (literal eta_ch, diagnostic); "
                         f"the default Lim preset is 'lim'")
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}") from exc


def with_consts(preset: Preset, **changes) -> Preset:
    """Return a copy of ``preset`` with some physical constants changed (label gets a suffix)."""
    suffix = ",".join(f"{k}={v:g}" for k, v in sorted(changes.items()))
    return Preset(name=f"{preset.name}[{suffix}]", flags=preset.flags,
                  consts=preset.consts._replace(**changes))


def with_flags(preset: Preset, name: str, **changes) -> Preset:
    return Preset(name=name, flags=preset.flags._replace(**changes), consts=preset.consts)

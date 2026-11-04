"""Physics presets. Every result produced by qkd_opt is labelled with one preset name.

Two presets are defined:

* ``lim_exact`` -- Lim, Curty, Walenta, Xu, Zbinden, PRA 89, 022307 (2014), arXiv:1311.7129,
  implemented literally from Eqs. (1)-(5) and the "dedicated fiber" error model in the
  Evaluation section:
    e_k = p_dc + e_mis [1 - exp(-eta_ch k)] + p_ap D_k / 2      (note: eta_ch, as printed)
    D_k = 1 - (1 - 2 p_dc) exp(-eta_sys k),  R_k = D_k (1 + p_ap)
    eps_sec = kappa * l  (solved by fixed-point iteration), kappa = 1e-15
    m_Z = sum_k m_{Z,k} (total Z-basis errors, Eq. 4)
* ``legacy`` -- the current ``src/qkd`` model, called unmodified
  (e_mis = 0.01, p_ap = 0, eps_sec = 1e-10 fixed, m_Z = sum of n_X terms, eta_sys in e_k).

``Flags`` are static (they select code paths, changing them recompiles); ``Consts`` are traced
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
    model: str = "lim"            # "lim" (src/qkd_opt/physics_lim.py) | "legacy_src" (src/qkd unmodified)
    emis_eta: str = "ch"          # transmittance in the e_mis term of e_k: "ch" (paper text) | "sys"
    mz_mode: str = "errors"       # m_Z in Eq. (4): "errors" = sum_k m_{Z,k} | "nx_legacy" = sum_k n_{X,k}
    eps_mode: str = "kappa"       # "kappa": eps_sec = kappa * l (fixed point) | "fixed": eps_sec = Consts.eps_sec
    afterpulse_in_rate: bool = True   # R_k = D_k (1 + p_ap) used for counts and N
    clamp_minus: bool = False     # clamp n^-, m^- at 0 (legacy behaviour); Lim's formulas do not
    n_eps_iter: int = 8           # monotone fixed-point iterations before log-bisection (eps_sec = kappa * l)


class Preset(NamedTuple):
    name: str
    flags: Flags
    consts: Consts


LIM_EXACT = Preset(
    name="lim_exact",
    flags=Flags(),
    consts=Consts(),
)

LEGACY = Preset(
    name="legacy",
    flags=Flags(model="legacy_src", emis_eta="sys", mz_mode="nx_legacy", eps_mode="fixed",
                afterpulse_in_rate=False, clamp_minus=True),
    consts=Consts(e_mis=0.01, p_ap=0.0, eps_sec=1e-10),
)

PRESETS = {p.name: p for p in (LIM_EXACT, LEGACY)}


def get_preset(name: str) -> Preset:
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

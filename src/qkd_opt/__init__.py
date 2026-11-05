"""qkd_opt: JAX-native parameter optimisation for finite-key decoy-state BB84 (Lim et al. 2014 bounds)."""
from .presets import (Consts, Flags, Preset, LIM, LIM_LITERAL, LEGACY, LEGACY_SRC, PRESETS, get_preset,
                      with_consts, with_flags)
from .objective import evaluate, key_rate, check_x64

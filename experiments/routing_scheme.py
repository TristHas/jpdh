#!/usr/bin/env python3
"""Sensitivity to the river-routing scheme (Figure 9d).

Everything at the default configuration except the routing kernel (the
linear-storage kernel of the default configuration is not repeated).  The
routing-parameter module reads the channel length, except for the Hayami
kernel, which takes the reach length from the network and reads the slope and
the drainage area.  Each kernel uses its bounds of ``config.KERNEL_BOUNDS``.

  irf_nash_cascade_L   Nash cascade
  irf_muskingum_L      Muskingum
  irf_pure_lag_L       pure lag (advection only), delay in [0, 0.5] day
  irf_hayami_SA        Hayami (diffusive wave)

3 seeds.

    python experiments/routing_scheme.py --seeds 0-2
"""
from _common import kernel_flags, launch

L, S, A = "channel_length_km", "log_slope", "log_upa"

CONFIGS = [
    ("irf_nash_cascade_L", kernel_flags("nash_cascade", [L])),
    ("irf_muskingum_L",    kernel_flags("muskingum", [L])),
    ("irf_pure_lag_L",     kernel_flags("pure_lag", [L])),
    ("irf_hayami_SA",      kernel_flags("hayami", [S, A])),
]

if __name__ == "__main__":
    launch("routing_scheme", CONFIGS, __doc__, seeds="0-2", base=["--save_series"])

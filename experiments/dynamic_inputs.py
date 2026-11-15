#!/usr/bin/env python3
"""Sensitivity to additional dynamic inputs (Figure 9c).

Each configuration adds MSM analysis fields (4-hour bins) to the default
precipitation and temperature inputs.  3 seeds; inputs in the dataset's
panels archive.

    python experiments/dynamic_inputs.py --seeds 0-2
"""
from _common import BASE_DYNAMIC, dyn, launch

WIND = ("msm_a_u_4h_bin", "msm_a_v_4h_bin")
SP, NCLD = "msm_a_sp_4h_bin", "msm_a_ncld_4h_bin"

CONFIGS = [
    ("extra_wind", dyn(BASE_DYNAMIC, *WIND)),
    ("extra_sp",   dyn(BASE_DYNAMIC, SP)),
    ("extra_ncld", dyn(BASE_DYNAMIC, NCLD)),
    ("extra_all",  dyn(BASE_DYNAMIC, *WIND, SP, NCLD)),
]

if __name__ == "__main__":
    launch("dynamic_inputs", CONFIGS, __doc__, seeds="0-2", base=["--save_series"])

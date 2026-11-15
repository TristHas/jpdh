#!/usr/bin/env python3
"""Sensitivity to the temporal resolution of the dynamic inputs (Figure 9b).

Precipitation and temperature binned at matching resolutions, from 24 hourly
bins to two 12-hour bins of the 12Z day (the default configuration uses 4-hour
bins and is not repeated).  3 seeds; inputs in the dataset's panels archive.

    python experiments/temporal_resolution.py --seeds 0-2
"""
from _common import dyn, launch

CONFIGS = [
    ("res_1h",  dyn("gauge_prcp_1h_bin", "msm_a_temp_1h_bin")),
    ("res_2h",  dyn("gauge_prcp_2h_bin", "msm_a_temp_2h_bin")),
    ("res_6h",  dyn("gauge_prcp_6h_bin", "msm_a_temp_6h_bin")),
    ("res_12h", dyn("gauge_prcp_12h_bin", "msm_a_temp_12h_bin")),
]

if __name__ == "__main__":
    launch("temporal_resolution", CONFIGS, __doc__, seeds="0-2", base=["--save_series"])

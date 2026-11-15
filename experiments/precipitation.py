#!/usr/bin/env python3
"""Sensitivity to the precipitation product (Figure 9a).

Everything at the default configuration except the precipitation input (the
in-situ product of the default configuration is not repeated):

  prcp_garadar   JMA gauge-adjusted radar
  prcp_radar     JMA radar composite
  prcp_msm       JMA MSM analysis
  prcp_era       ERA5-Land

3 seeds.  The inputs are in the dataset's panels archive
(``python scripts/download_data.py --panels``).

    python experiments/precipitation.py --seeds 0-2
"""
from _common import BASE_TEMP, dyn, launch

CONFIGS = [
    ("prcp_garadar", dyn("garadar_prcp_4h_bin", BASE_TEMP)),
    ("prcp_radar",   dyn("radar_prcp_4h_bin", BASE_TEMP)),
    ("prcp_msm",     dyn("msm_a_prcp_4h_bin", BASE_TEMP)),
    ("prcp_era",     dyn("era_prcp_4h", BASE_TEMP)),
]

if __name__ == "__main__":
    launch("precipitation", CONFIGS, __doc__, seeds="0-2", base=["--save_series"])

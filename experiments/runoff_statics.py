#!/usr/bin/env python3
"""Runoff static descriptors used in isolation (Figure 9e).

Each run replaces the default descriptor set (topography, land use, MSM
seasonality) with a single group, which makes the panel a statement about
that group's own information content rather than its marginal contribution.
3 seeds.

    python experiments/runoff_statics.py --seeds 0-2
"""
from _common import launch

GROUPS = [
    ("topo",           "topo"),
    ("land_use",       "land_use"),
    ("land_cover_pct", "land_cover_pct"),
    ("terrain",        "terrain"),
    ("water_storage",  "water_storage"),
    ("soil",           "soil_properties,soil_texture"),
    ("population",     "population,human_impact"),
    ("wet",            "wet_cl_smj"),
    ("era_sd",         "era_sd"),
    ("era_swvl1",      "era_swvl1"),
    ("snw_monthly",    "snw_pc_monthly"),
    ("pre_monthly",    "pre_mm_monthly"),
    ("aet_monthly",    "aet_mm_monthly"),
]

CONFIGS = [(f"runstat_{n}", ["--static_runoff_vars", v]) for n, v in GROUPS]

if __name__ == "__main__":
    launch("runoff_statics", CONFIGS, __doc__, seeds="0-2", base=["--save_series"])

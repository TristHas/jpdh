"""Variable dictionaries, the default configuration and the dataset location."""
from __future__ import annotations

import os
from pathlib import Path


def _bins(name: str, n: int) -> list[str]:
    return [f"{name}_{i}" for i in range(n)]


#: Dynamic (forcing) inputs: a key expands to the variables of
#: ``daily/forcing.zarr`` (default inputs) or ``daily/forcing_panels.zarr``
#: (inputs of the sensitivity experiments).  Products: ``gauge`` (in-situ
#: gauge-interpolated precipitation), ``garadar`` (JMA gauge-adjusted radar),
#: ``radar`` (JMA C-band radar), ``msm_a`` (JMA MSM analysis), ``era``
#: (ERA5-Land).  ``<var>_<N>h_bin_<i>`` is the i-th N-hour bin of the 12Z day.
DYNAMIC_VAR_DICT = {
    # default configuration: in-situ precipitation and MSM temperature, 4-h bins
    'gauge_prcp_4h_bin':   _bins('gauge_prcp_4h_bin', 6),
    'msm_a_temp_4h_bin':   _bins('msm_a_temp_4h_bin', 6),

    # precipitation products
    'garadar_prcp_4h_bin': _bins('garadar_prcp_4h_bin', 6),
    'radar_prcp_4h_bin':   _bins('radar_prcp_4h_bin', 6),
    'msm_a_prcp_4h_bin':   _bins('msm_a_prcp_4h_bin', 6),
    'era_prcp_4h':         _bins('era_prcp_4h_bin', 6),
    'era_temp_4h_bin':     _bins('era_temp_4h_bin', 6),

    # temporal resolution of the inputs
    'gauge_prcp':          ['gauge_prcp'],
    'gauge_prcp_12h_bin':  _bins('gauge_prcp_12h_bin', 2),
    'gauge_prcp_6h_bin':   _bins('gauge_prcp_6h_bin', 4),
    'gauge_prcp_2h_bin':   _bins('gauge_prcp_2h_bin', 12),
    'gauge_prcp_1h_bin':   _bins('gauge_prcp_1h_bin', 24),
    'msm_a_temp':          ['msm_a_temp'],
    'msm_a_temp_12h_bin':  _bins('msm_a_temp_12h_bin', 2),
    'msm_a_temp_6h_bin':   _bins('msm_a_temp_6h_bin', 4),
    'msm_a_temp_2h_bin':   _bins('msm_a_temp_2h_bin', 12),
    'msm_a_temp_1h_bin':   _bins('msm_a_temp_1h_bin', 24),

    # additional MSM fields
    'msm_a_u_4h_bin':      _bins('msm_a_u_4h_bin', 6),
    'msm_a_v_4h_bin':      _bins('msm_a_v_4h_bin', 6),
    'msm_a_sp_4h_bin':     _bins('msm_a_sp_4h_bin', 6),
    'msm_a_ncld_4h_bin':   _bins('msm_a_ncld_4h_bin', 6),
}


#: Runoff static descriptors (columns of ``statics/runoff_statics.parquet``),
#: grouped.  HydroATLAS attribute names (Linke et al., 2019) where applicable.
STATIC_VAR_DICT = {
    # ERA5-Land monthly climatologies
    'era_sd':    _bins('era_sd', 12),
    'era_swvl1': _bins('era_swvl1', 12),
    'era_swvl2': _bins('era_swvl2', 12),
    'era_swvl3': _bins('era_swvl3', 12),
    'era_swvl4': _bins('era_swvl4', 12),

    # Hydrology / river / terrain
    'river_flow':    ['dis_m3_pyr', 'dis_m3_pmn', 'dis_m3_pmx', 'run_mm_syr'],
    'inundation':    ['inu_pc_smn', 'inu_pc_umn', 'inu_pc_smx', 'inu_pc_umx', 'inu_pc_slt', 'inu_pc_ult'],
    'water_storage': ['lka_pc_sse', 'lka_pc_use', 'lkv_mc_usu', 'rev_mc_usu', 'dor_pc_pva', 'gwt_cm_sav'],
    'river_network': ['ria_ha_ssu', 'ria_ha_usu', 'riv_tc_ssu', 'riv_tc_usu'],
    'terrain':       ['ele_mt_sav', 'ele_mt_uav', 'ele_mt_smn', 'ele_mt_smx', 'slp_dg_sav', 'slp_dg_uav', 'sgr_dk_sav'],
    'climate_zone':  ['clz_cl_smj', 'cls_cl_smj'],

    # Temperature
    'tmp_dc': ['tmp_dc_syr', 'tmp_dc_uyr', 'tmp_dc_smn', 'tmp_dc_smx'],
    'tmp_dc_monthly': [f'tmp_dc_s{i:02d}' for i in range(1, 13)],

    # Precipitation
    'pre_mm_syr': ['pre_mm_syr', 'pre_mm_uyr'],
    'pre_mm_monthly': [f'pre_mm_s{i:02d}' for i in range(1, 13)],

    # PET
    'pet_mm_yr': ['pet_mm_syr', 'pet_mm_uyr'],
    'pet_mm_monthly': [f'pet_mm_s{i:02d}' for i in range(1, 13)],

    # AET
    'aet_mm_yr': ['aet_mm_syr', 'aet_mm_uyr'],
    'aet_mm_monthly': [f'aet_mm_s{i:02d}' for i in range(1, 13)],

    # Aridity
    'ari_ix': ['ari_ix_sav', 'ari_ix_uav'],

    # CMI
    'cmi_ix_yr': ['cmi_ix_syr', 'cmi_ix_uyr'],
    'cmi_ix_monthly': [f'cmi_ix_s{i:02d}' for i in range(1, 13)],

    # Snow
    'snw_pc_syr': ['snw_pc_syr', 'snw_pc_uyr', 'snw_pc_smx'],
    'snw_pc_monthly': [f'snw_pc_s{i:02d}' for i in range(1, 13)],

    # Glacier
    'glc_cl_smj': ['glc_cl_smj'],
    'glc_pc_s': [f'glc_pc_s{i:02d}' for i in range(1, 23)],
    'glc_pc_u': [f'glc_pc_u{i:02d}' for i in range(1, 23)],

    # Potential natural vegetation
    'pnv_cl_smj': ['pnv_cl_smj'],
    'pnv_pc_s': [f'pnv_pc_s{i:02d}' for i in range(1, 16)],
    'pnv_pc_u': [f'pnv_pc_u{i:02d}' for i in range(1, 16)],

    # Wetlands
    'wet_cl_smj': ['wet_cl_smj', 'wet_pc_sg1', 'wet_pc_ug1', 'wet_pc_sg2', 'wet_pc_ug2'],
    'wet_pc_s': [f'wet_pc_s{i:02d}' for i in range(1, 10)],
    'wet_pc_u': [f'wet_pc_u{i:02d}' for i in range(1, 10)],

    # Land cover / land use
    'land_cover_pct': ['for_pc_sse', 'for_pc_use', 'crp_pc_sse', 'crp_pc_use',
                       'pst_pc_sse', 'pst_pc_use', 'ire_pc_sse', 'ire_pc_use',
                       'gla_pc_sse', 'gla_pc_use', 'prm_pc_sse', 'prm_pc_use',
                       'pac_pc_sse', 'pac_pc_use'],
    'biome_class':    ['tbi_cl_smj', 'tec_cl_smj', 'fmh_cl_smj', 'fec_cl_smj'],

    # Soil
    'soil_texture':    ['cly_pc_sav', 'cly_pc_uav', 'slt_pc_sav', 'slt_pc_uav', 'snd_pc_sav', 'snd_pc_uav'],
    'soil_properties': ['soc_th_sav', 'soc_th_uav', 'lit_cl_smj', 'kar_pc_sse', 'kar_pc_use', 'ero_kh_sav', 'ero_kh_uav'],
    'swc_pc':          ['swc_pc_syr', 'swc_pc_uyr'] + [f'swc_pc_s{i:02d}' for i in range(1, 13)],

    # Population / infrastructure / economy
    'population':   ['pop_ct_ssu', 'pop_ct_usu', 'ppd_pk_sav', 'ppd_pk_uav'],
    'human_impact': ['urb_pc_sse', 'urb_pc_use', 'nli_ix_sav', 'nli_ix_uav',
                     'rdd_mk_sav', 'rdd_mk_uav', 'hft_ix_s93', 'hft_ix_u93', 'hft_ix_s09', 'hft_ix_u09'],
    'socioeconomy': ['gad_id_smj', 'gdp_ud_sav', 'gdp_ud_ssu', 'gdp_ud_usu', 'hdi_ix_sav'],

    # Land use (JAXA) and topography (J-FlwDir)
    'land_use': ['Water bodies', 'Built-up', 'Paddy field', 'Cropland',
                 'Grassland', 'DBF', 'DNF', 'EBF', 'ENF', 'Bare'],
    'topo': ['elv_mean', 'elv_std', 'area'],

    # Seasonal indicators
    'season_gauge': [f'season_gauge_{i}' for i in range(1, 13)],
    'season_msm': [f'season_msm_{i}' for i in range(1, 13)],

    # Snow summary
    'msm_snow_avg': ['msm_snow_avg'],
    'gauge_snow_avg': ['gauge_snow_avg'],
}


def expand_dynamic_keys(keys):
    """Expand a list of DYNAMIC_VAR_DICT keys into a flat variable list."""
    out = []
    for k in keys:
        if k not in DYNAMIC_VAR_DICT:
            raise KeyError(f"Unknown dynamic key: {k!r}. Available: {list(DYNAMIC_VAR_DICT)}")
        out.extend(DYNAMIC_VAR_DICT[k])
    return out


def expand_static_keys(keys):
    """Expand a list of STATIC_VAR_DICT keys into a flat variable list."""
    out = []
    for k in keys:
        if k not in STATIC_VAR_DICT:
            raise KeyError(f"Unknown static key: {k!r}. Available: {list(STATIC_VAR_DICT)}")
        out.extend(STATIC_VAR_DICT[k])
    return out


# ---------------------------------------------------------------------------
# Default configuration (Table 2 of the paper)
# ---------------------------------------------------------------------------

DEFAULT_DYNAMIC_KEYS = ['gauge_prcp_4h_bin', 'msm_a_temp_4h_bin']
DEFAULT_STATIC_RUNOFF_KEYS = ['topo', 'land_use', 'season_msm']
#: routing kernel and the descriptors its parameter MLP reads
DEFAULT_IRF = "linear_storage"
DEFAULT_ROUTING_STATIC_VAR = ["channel_length_km"]
DEFAULT_ROUTING_LR = 1e-4
#: parameter bounds per kernel, in the hourly units of DiffHydro's
#: ``PARAMS_BOUNDS`` (linear-storage / Nash-cascade tau and Muskingum k in
#: hours, pure-lag delay in daily steps); ``train.py --routing_bounds``
#: overrides them
KERNEL_BOUNDS = {
    "linear_storage": "tau=0.01,9.9",
    "nash_cascade":   "tau=0.01,9.9",
    "pure_lag":       "delay=0,0.5",
    "muskingum":      "k=0.01,6;x=0,0.5",
    "hayami":         "D=0.02,200;c=0.5,50",
}
#: initial value of every routing parameter: "low" = sigmoid(-3), 4.7 % of the range
DEFAULT_ROUTING_INIT = "low"


# ---------------------------------------------------------------------------
# Dataset location
# ---------------------------------------------------------------------------

#: the dataset root holds network/, statics/, daily/, splits/ (and qc/);
#: ``JPDH_DATA_ROOT`` overrides it
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def data_root() -> Path:
    """Root of the dataset (``$JPDH_DATA_ROOT``, default ``<repository>/data``)."""
    return Path(os.environ.get("JPDH_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser()

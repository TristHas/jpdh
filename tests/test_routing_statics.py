"""The routing-descriptor table: raw channel length from the node table,
transform-then-standardise, headwater NaN -> 0."""
import numpy as np
import pandas as pd
import pytest

from data import routing_static_table


@pytest.fixture
def tables(tmp_path):
    idx = pd.Index([10, 11, 12, 13], name="node")
    rs = pd.DataFrame({"log_upa": [1.0, 2.0, 3.0, np.nan],
                       "log_slope": [-3.0, -2.0, -1.0, np.nan],
                       "dist": [10.0, 20.0, 30.0, 40.0]}, index=idx)
    (tmp_path / "routing_statics.parquet").parent.mkdir(exist_ok=True)
    rs.to_parquet(tmp_path / "routing_statics.parquet")
    nodes = pd.DataFrame({"channel_length_km": [1.0, 2.0, 4.0, 8.0]},
                         index=pd.Index([13, 12, 11, 10]))     # deliberately shuffled
    return tmp_path, nodes


def test_channel_length_comes_from_the_node_table_and_is_linear(tables):
    d, nodes = tables
    t = routing_static_table(d, nodes, ["channel_length_km", "log_upa", "log_slope"])
    assert list(t.columns) == ["channel_length_km", "log_upa", "log_slope"]
    raw = nodes["channel_length_km"].reindex(t.index)
    z = (raw - raw.mean()) / raw.std()
    np.testing.assert_allclose(t["channel_length_km"].values, z.values)  # aligned by node id, linear
    assert abs(t["channel_length_km"].mean()) < 1e-12 and abs(t["channel_length_km"].std() - 1) < 1e-12


def test_headwater_nan_becomes_zero_after_standardising(tables):
    d, nodes = tables
    t = routing_static_table(d, nodes, ["log_upa", "log_slope"])
    assert t.loc[13].tolist() == [0.0, 0.0]
    assert abs(t.loc[[10, 11, 12], "log_upa"].mean()) < 1e-12   # z-scored over the non-NaN rows


def test_unknown_descriptor_is_refused(tables):
    d, nodes = tables
    with pytest.raises(KeyError):
        routing_static_table(d, nodes, ["chan_len"])

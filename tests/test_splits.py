"""Splits files (minimal and full schemas) and time-block resolution."""
import json

import pandas as pd
import pytest

from data import (load_splits, load_time_blocks, read_splits_file, sha256_file,
                  splits_from_payload, time_blocks_path)

FOLDS = [dict(train=[1, 2, 3], val=[4], test=[4, 5]),
         dict(train=[1, 4, 5], val=[2], test=[2, 3])]


def _write(tmp_path, name, **extra):
    p = tmp_path / name
    p.write_text(json.dumps(dict(n_folds=2, folds=FOLDS,
                                 gauges_with_discharge=[1, 2, 3, 4, 5], **extra)))
    return p


def test_minimal_schema(tmp_path):
    p = _write(tmp_path, "pub_folds_10.json")
    payload = read_splits_file(p)
    assert payload["_sha256"] == sha256_file(p) and payload["_path"] == str(p.resolve())
    folds, gauges = load_splits(path=p)
    assert folds == [[{1, 2, 3}, {4}, {4, 5}], [{1, 4, 5}, {2}, {2, 3}]]
    assert gauges == [1, 2, 3, 4, 5]
    assert time_blocks_path(p, payload) is None            # nothing next to it


def test_full_schema_and_time_blocks(tmp_path):
    blocks = tmp_path / "time_blocks.json"
    blocks.write_text(json.dumps({"B1": ["2009-01-01", "2012-12-31"],
                                  "B2": ["2013-01-01", "2016-12-31"],
                                  "B3": ["2017-01-01", "2020-12-31"],
                                  "label": "inclusive dates"}))
    p = _write(tmp_path, "pub_folds_10.json",
               evaluation_set=[1, 2, 3, 4, 5], training_pool=[1, 2, 3, 4, 5],
               min_valid_days=364,
               time_blocks_file="time_blocks.json")
    payload = read_splits_file(p)
    assert payload["min_valid_days"] == 364
    assert splits_from_payload(payload)[0][2] == {4, 5}
    assert time_blocks_path(p, payload) == blocks                    # key, relative
    assert time_blocks_path(p, payload, override=tmp_path / "x.json") == tmp_path / "x.json"
    assert time_blocks_path(p, dict(payload, time_blocks_file=None)) == blocks   # sibling
    tb = load_time_blocks(blocks)
    assert set(tb) == {"B1", "B2", "B3"}
    assert tb["B3"] == (pd.Timestamp("2017-01-01"), pd.Timestamp("2020-12-31"))


def test_not_a_splits_file(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"folds": [{"train": [1]}]}))
    with pytest.raises(ValueError):
        read_splits_file(p)

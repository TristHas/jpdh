from pathlib import Path

from provenance import git_describe, lib_states, run_provenance


def test_lib_states_are_describable():
    libs = lib_states()
    assert set(libs) == {"diffhydro", "diffroute", "xtensor", "jpdh"}
    for name, state in libs.items():
        assert isinstance(state, str) and state, name
    assert git_describe("/") == "unknown"
    prov = run_provenance()
    assert prov["libs"] == libs and prov["started_at"]

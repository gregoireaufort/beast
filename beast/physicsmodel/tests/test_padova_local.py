import numpy as np
import pytest
from astropy.table import Table

from beast.physicsmodel.stars import isochrone
from beast.physicsmodel.stars.padova_local import (
    MissingIsochroneError,
    PadovaLocalDatabase,
    PadovaQuerySpec,
)


def _raw_parsec_table(z=0.001):
    return Table(
        {
            "logAge": np.array([6.0, 6.0, 7.0]),
            "logTe": np.array([4.1, 4.0, 3.9]),
            "Mini": np.array([10.0, 9.0, 5.0]),
            "Mass": np.array([9.8, 8.8, 4.9]),
            "label": np.array([1, 2, 3]),
            "Z": np.array([z, z, z]),
            "Zini": np.array([z, z, z]),
            "logL": np.array([3.0, 2.8, 2.0]),
            "logg": np.array([4.0, 4.1, 4.2]),
            "mbolmag": np.array([-2.0, -1.8, -1.0]),
            "U": np.array([1.0, 1.0, 1.0]),
            "X": np.array([0.7, 0.7, 0.7]),
        }
    )


def test_padova_query_spec_key_is_stable_and_sensitive():
    spec = PadovaQuerySpec.for_beast_t_isochrones(
        5.0, 10.2, 0.1, 0.001, modeltype="parsec12s_r14"
    )
    same = PadovaQuerySpec.for_beast_t_isochrones(
        6.0, 10.13, 0.1, 0.001, modeltype="parsec12s_r14"
    )
    different = PadovaQuerySpec.for_beast_t_isochrones(
        6.0, 10.13, 0.1, 0.002, modeltype="parsec12s_r14"
    )

    assert spec.key() == same.key()
    assert spec.key() != different.key()
    assert spec.as_dict()["logtmin"] == 6.0
    assert spec.as_dict()["logtmax"] == 10.13


def test_padova_local_returns_same_cleaned_columns_as_web(tmp_path):
    z = 0.001
    spec = PadovaQuerySpec.for_beast_t_isochrones(
        6.0, 7.0, 1.0, z, modeltype="parsec12s_r14"
    )
    db = PadovaLocalDatabase(tmp_path)
    db.write_raw_table(spec, _raw_parsec_table(z=z), raw_bytes=b"raw cmd table")

    web = isochrone.PadovaWeb(modeltype="parsec12s_r14")
    expected = web._clean_cols(_raw_parsec_table(z=z).copy())
    expected = web._filter_iso_points(expected, filterPMS=False, filterBad=False)

    local = isochrone.PadovaLocal(
        database_path=tmp_path, modeltype="parsec12s_r14", offline_strict=True
    )
    actual = local._get_t_isochrones(6.0, 7.0, 1.0, z)

    assert actual.colnames == expected.colnames
    for colname in expected.colnames:
        np.testing.assert_allclose(actual[colname], expected[colname], rtol=0, atol=0)
    assert actual.header["NAME"] == "PadovaCMD Isochrones: parsec12s_r14"


def test_padova_local_offline_miss_does_not_call_remote(tmp_path, monkeypatch):
    def fail_remote(*args, **kwargs):
        raise AssertionError("remote query should not be called in strict offline mode")

    monkeypatch.setattr(isochrone.parsec, "get_t_isochrones", fail_remote)
    local = isochrone.PadovaLocal(
        database_path=tmp_path, modeltype="parsec12s_r14", offline_strict=True
    )

    with pytest.raises(MissingIsochroneError):
        local._get_t_isochrones(6.0, 7.0, 1.0, 0.001)


def test_padova_local_non_strict_populates_cache_from_mock_remote(tmp_path, monkeypatch):
    z = 0.001

    def mock_remote(logtmin, logtmax, dlogt, metal, model=None):
        assert (logtmin, logtmax, dlogt, metal, model) == (
            6.0,
            7.0,
            1.0,
            z,
            "parsec12s_r14",
        )
        return _raw_parsec_table(z=metal)

    monkeypatch.setattr(isochrone.parsec, "get_t_isochrones", mock_remote)
    local = isochrone.PadovaLocal(
        database_path=tmp_path, modeltype="parsec12s_r14", offline_strict=False
    )

    first = local._get_t_isochrones(6.0, 7.0, 1.0, z)
    assert "M_ini" in first.colnames

    strict_local = isochrone.PadovaLocal(
        database_path=tmp_path, modeltype="parsec12s_r14", offline_strict=True
    )
    second = strict_local._get_t_isochrones(6.0, 7.0, 1.0, z)
    np.testing.assert_allclose(second["M_ini"], first["M_ini"], rtol=0, atol=0)

import numpy as np
from astropy.table import Table

from beast.physicsmodel.grid import SpectralGrid
from beast.physicsmodel import creategrid
from beast.physicsmodel.dust import extinction


def _tiny_spectral_grid():
    lamb = np.linspace(1200.0, 25000.0, 400)
    seds = np.vstack(
        [
            1.0e-9 * (lamb / lamb.mean()) ** -1.0,
            1.5e-9 * (lamb / lamb.mean()) ** -0.3,
            0.8e-9 * (lamb / lamb.mean()) ** 0.7,
        ]
    )
    grid = Table(
        {
            "distance": np.array([10.0, 10.0, 10.0]),
            "weight": np.ones(3),
            "prior_weight": np.ones(3),
            "grid_weight": np.ones(3),
            "logA": np.array([6.0, 7.0, 8.0]),
            "M_ini": np.array([1.0, 2.0, 3.0]),
            "Z": np.array([0.001, 0.001, 0.002]),
        }
    )
    return SpectralGrid(lamb, seds=seds, grid=grid, backend="memory")


def test_fast_extinguished_grid_matches_original_with_fa_and_spectral_properties():
    specgrid = _tiny_spectral_grid()
    filters = ["HST_WFC3_F275W", "HST_ACS_WFC_F475W", "HST_WFC3_F160W"]
    extlaw = extinction.Generalized_RvFALaw(
        ALaw=extinction.Generalized_DustExt(curve="F19"),
        BLaw=extinction.Generalized_DustExt(curve="G03_SMCBar"),
    )
    avs = np.array([0.0, 0.5])
    rvs = np.array([2.5, 3.1])
    fAs = np.array([0.5, 1.0])
    add_props = {"filternames": filters}

    original = next(creategrid.make_extinguished_grid(
        specgrid,
        filters,
        extlaw,
        avs,
        rvs,
        fAs,
        add_spectral_properties_kwargs=dict(add_props),
    ))
    fast = next(creategrid.make_extinguished_grid_fast(
        specgrid,
        filters,
        extlaw,
        avs,
        rvs,
        fAs,
        add_spectral_properties_kwargs=dict(add_props),
    ))

    np.testing.assert_allclose(fast.lamb, original.lamb, rtol=0, atol=0)
    np.testing.assert_allclose(fast.seds, original.seds, rtol=5e-13, atol=0)

    for colname in original.grid.colnames:
        if original.grid[colname].dtype.kind in "fiu":
            np.testing.assert_allclose(
                fast.grid[colname], original.grid[colname], rtol=5e-13, atol=0
            )
        else:
            assert np.all(fast.grid[colname] == original.grid[colname])

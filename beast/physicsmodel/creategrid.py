"""
Create extinguished grid more segmented dealing with large grids with enough
memory

All functions are now transformed into generators. As a result, any function
allows computation of a grid in an arbitrary number of chunks. This offers the
possibility to generate grids that cannot fit in memory.


.. note::

    * dependencies have also been updated accordingly.

    * likelihood computations need to be updated to allow computations even if
      the full grid does not fit in memory
"""

import numpy as np
import copy

from astropy import units
from tqdm import tqdm

from beast.physicsmodel.stars import stellib
from beast.physicsmodel.grid import SpectralGrid, SEDGrid
from beast.physicsmodel.grid_and_prior_weights import compute_av_rv_fA_prior_weights

from beast.physicsmodel.grid_weights import compute_grid_weights

from astropy.table import Table
from beast.tools.helpers import generator
from beast.tools import helpers
from beast.tools.profiling import profile_stage

from beast.observationmodel.noisemodel import absflux_covmat
from beast.observationmodel import phot

__all__ = [
    "gen_spectral_grid_from_stellib_given_points",
    "make_extinguished_grid",
    "make_extinguished_grid_fast",
    "add_spectral_properties",
    "calc_absflux_cov_matrices",
]


@generator
def gen_spectral_grid_from_stellib_given_points(
    osl, pts, bounds=dict(dlogT=0.1, dlogg=0.3), chunksize=0
):
    """
    Generator that reinterpolates a given stellar spectral library on to
       an Isochrone grid

    It will iterate over a list of `pts` points and generate
       `chunksize` models until all the list of points is processed

    Parameters
    ----------
    osl: stellib.stellib
        a stellar library

    pts: dict like structure of points
        dictionary like or named data structure of points to interpolate at
        must contain logg, logT, logL, and Z

    bounds:  dict, optional (default={dlogT:0.1, dlogg:0.3})
        sensitivity to extrapolation (see grid.get_stellib_boundaries)

    chunksize: int, optional (default=0)
        number of models to generate at each cycle.
        If default <= 0, all models will be returned at once.

    Returns
    -------
    g: SpectralGrid
        Spectral grid (in memory) containing the requested list of stars
        and associated spectra
    """

    helpers.type_checker("osl", osl, stellib.Stellib)

    if chunksize <= 0:
        with profile_stage("physics: stellar spectral interpolation"):
            yield osl.gen_spectral_grid_from_given_points(pts, bounds=bounds)
    else:
        try:
            # Yield successive n-sized chunks from l, assuming we can take
            # slices of the iterator
            for chunk_slice in helpers.chunks(list(range(len(pts))), chunksize):
                chunk_pts = pts[chunk_slice]
                with profile_stage("physics: stellar spectral interpolation"):
                    yield osl.gen_spectral_grid_from_given_points(
                        chunk_pts, bounds=bounds
                    )
        except Exception as e:
            # chunks may not work on this as pts is most likely a Table
            print(e)
            for chunk_pts in helpers.chunks(pts, chunksize):
                with profile_stage("physics: stellar spectral interpolation"):
                    yield osl.gen_spectral_grid_from_given_points(
                        chunk_pts, bounds=bounds
                    )


def _make_dust_fA_valid_points_generator(it, min_Rv, max_Rv, Rv_B):
    """
    compute the allowed points based on the R(V) versus f_A plane
    duplicates effort for all A(V) values, but it is quick compared to
    other steps

    Parameters
    ----------
    it: an iterable
        an initial sequence of points that will be trimmed to only valid ones

    min_Rv: float
        lower Rv limit

    max_Rv: float
        upper Rv limit

    Rv_B: float
        value of R(V) from the user-defined SMC-like extinction law

    Returns
    -------
    npts: int
        the actual number of valid points

    pts: generator
        a generator that only produce valid points
    """
    itn = copy.copy(it)
    npts = 0

    def is_valid(ak, rk, fk):
        return (
            fk / max_Rv + (1.0 - fk) / Rv_B
            <= 1.0 / rk
            <= fk * 1.0 / min_Rv + (1.0 - fk) / Rv_B
        )

    # explore the full list once
    # not very time consuming
    for ak, rk, fk in itn:
        if is_valid(ak, rk, fk):
            npts += 1

    # make the iterator
    pts = (
        (float(ak), float(rk), float(fk)) for ak, rk, fk in it if is_valid(ak, rk, fk)
    )

    return npts, pts


def _filter_integration_matrix(lamb, filters, absFlux=True):
    """
    Build the linear operator used by phot.extractSEDs.

    The resulting matrix has shape (n_wavelength, n_filters), so
    ``spectra @ matrix`` is numerically equivalent to extracting SEDs one
    filter at a time with the existing trapezoid integration.
    """
    lamb = np.asarray(lamb)
    matrix = np.zeros((lamb.size, len(filters)), dtype=float)
    cls = np.empty(len(filters), dtype=float)

    for f_indx, filt in enumerate(filters):
        xl = filt.transmit > 0.0
        if not np.any(xl):
            cls[f_indx] = filt.cl
            continue

        x = lamb[xl]
        trap_weights = np.empty(x.size, dtype=float)
        if x.size == 1:
            trap_weights[0] = 0.0
        else:
            trap_weights[0] = 0.5 * (x[1] - x[0])
            trap_weights[-1] = 0.5 * (x[-1] - x[-2])
            if x.size > 2:
                trap_weights[1:-1] = 0.5 * (x[2:] - x[:-2])

        weights = x * filt.transmit[xl] * trap_weights / filt.lT
        if absFlux:
            weights = weights / phot.distc
        matrix[xl, f_indx] = weights
        cls[f_indx] = filt.cl

    return cls, matrix


def _log_sed_columns(seds):
    log_seds = np.full(seds.shape, -100.0, dtype=float)
    good = seds > 0
    log_seds[good] = np.log10(seds[good])
    return log_seds


def apply_distance_grid(specgrid, distances, redshift=0):
    """
    Distances are applied to the spectral grid by copying the grid and
    applying a scaling factor.

    Parameters
    ----------

    project: str
        project name

    specgrid: grid.SpectralGrid object
        spectral grid to transform

    distances: list of float
        Distances at which models should be shifted
        0 means absolute magnitude.
        Expecting pc units

    redshift: float
        Redshift to which wavelengths should be shifted
        Default is 0 (rest frame)
    """
    g0 = specgrid

    # Current length of the grid
    N0 = len(g0.grid)
    N = N0 * len(distances)

    # Make singleton list if a single distance is given
    if not hasattr(distances, "__iter__"):
        _distances = [distances]
    else:
        _distances = distances

    # Add distance column if multiple distances are specified
    cols = {}
    cols["distance"] = np.zeros(N, dtype=float)

    # Existing columns
    keys0 = list(g0.keys())
    for key in keys0:
        cols[key] = np.zeros(N, dtype=float)

    n_sed_points = g0.seds.shape[1]
    new_seds = np.zeros((N, n_sed_points), dtype=float)

    for count, distance in enumerate(tqdm(_distances, desc="Distance grid")):

        # The range where the current distance points will live
        distance_slice = slice(N0 * count, N0 * (count + 1))

        # The seds default to 10 pc.
        # Therefore, scale them with (d / (10 pc))**(-2).
        distance_pc = distance.to(units.pc).value
        new_seds[distance_slice, :] = g0.seds / (0.1 * distance_pc) ** 2

        # Fill in the distance in the distance column
        cols["distance"][distance_slice] = distance_pc

        # Copy the old columns
        for key in keys0:
            cols[key][distance_slice] = g0.grid[key]

    # apply redshift
    g0.lamb = g0.lamb * (1.0 + redshift)

    # New object
    g = SpectralGrid(g0.lamb, seds=new_seds, grid=Table(cols), backend="memory")
    return g


@generator
def make_extinguished_grid(
    spec_grid,
    filter_names,
    extLaw,
    avs,
    rvs,
    fAs=None,
    av_prior_model={"name": "flat"},
    rv_prior_model={"name": "flat"},
    fA_prior_model={"name": "flat"},
    chunksize=0,
    add_spectral_properties_kwargs=None,
    absflux_cov=False,
    filterLib=None,
):
    """
    Extinguish spectra and extract an SEDGrid through given series of filters
    (all wavelengths in stellar SEDs and filter response functions are assumed
    to be in Angstroms)

    Parameters
    ----------
    spec_grid: string or grid.SpectralGrid
        if string:
        spec_grid is the filename to the grid file with stellar spectra
        the backend to load this grid will be the minimal invasive: 'HDF'
        if possible, 'cache' otherwise.

        if not a string, expecting the corresponding SpectralGrid instance
        (backend already setup)

    filter_names: list
        list of filter names according to the filter lib

    Avs: sequence
        Av values to iterate over

    av_prior_model: list
        list including prior model name and parameters

    Rvs: sequence
        Rv values to iterate over

    rv_prior_model: list
        list including prior model name and parameters

    fAs: sequence (optional)
        f_A values to iterate over
        f_A can be omitted if the extinction Law does not use it or allow
        fixed values

    fA_prior_model: list
        list including prior model name and parameters

    chunksize: int, optional (default=0)
        number of extinction model variations to generate at each cycle.
        Note that this means len(spec_grid * chunksize)
        If default <= 0, all models will be returned at once.

    filterLib:  str
        full filename to the filter library hd5 file

    add_spectral_properties_kwargs: dict
        keyword arguments to call :func:`add_spectral_properties` at each
        iteration to add model properties from the spectra into the grid
        property table

    asbflux_cov: boolean
        set to calculate the absflux covariance matrices for each model
        (can be very slow!!!  But it is the right thing to do)

    Returns
    -------
    g: grid.SpectralGrid
        final grid of reddened SEDs and models
    """
    # Check inputs
    # ============
    # get the stellar grid (no dust yet)
    # if string is provided try to load the most memory efficient backend
    # otherwise use a cache-type backend (load only when needed)
    if isinstance(spec_grid, str):
        ext = spec_grid.split(".")[-1]
        if ext in ["hdf", "hd5", "hdf5"]:
            g0 = SpectralGrid(spec_grid, backend="disk")
        else:
            g0 = SpectralGrid(spec_grid, backend="cache")
    else:
        helpers.type_checker("spec_grid", spec_grid, SpectralGrid)
        g0 = spec_grid

    # Tag fA usage
    if fAs is None:
        with_fA = False
    else:
        with_fA = True

    # get the min/max R(V) values necessary for the grid point definition
    min_Rv = min(rvs)
    max_Rv = max(rvs)
    Rv_B = extLaw.BLaw.Rv  # R(V) for the SMC-like extinction curve

    # Create the sampling mesh
    # ========================
    # basically the dot product from all input 1d vectors
    # setup integration over the full dust parameter grid

    if with_fA:

        it = np.nditer(np.ix_(avs, rvs, fAs))
        niter = np.size(avs) * np.size(rvs) * np.size(fAs)
        npts, pts = _make_dust_fA_valid_points_generator(it, min_Rv, max_Rv, Rv_B)

        # Pet the user
        print(
            """number of initially requested points = {0:d}
              number of valid points = {1:d} (based on restrictions in R(V)
                 versus f_A plane)
              """.format(
                niter, npts
            )
        )

        if npts == 0:
            raise AttributeError("No valid points")
    else:
        it = np.nditer(np.ix_(avs, rvs))
        npts = np.size(avs) * np.size(rvs)
        pts = ((float(ak), float(rk)) for ak, rk in it)

    # Generate the Grid
    # =================
    N0 = len(g0.grid)
    N = N0 * npts

    if chunksize <= 0:
        print("Generating a final grid of {0:d} points".format(N))
    else:
        print(
            "Generating a final grid of {0:d} points in {1:d} pieces".format(
                N, int(float(N0) / chunksize + 1.0)
            )
        )

    if chunksize <= 0:
        chunksize = npts

    if add_spectral_properties_kwargs is not None:
        nameformat = add_spectral_properties_kwargs.pop("nameformat", "{0:s}") + "_wd"

    for chunk_pts in helpers.chunks(pts, chunksize):
        # iter over chunks of models

        # setup chunk outputs
        cols = {"Av": np.zeros(N, dtype=float), "Rv": np.zeros(N, dtype=float)}

        if with_fA:
            cols["Rv_A"] = np.zeros(N, dtype=float)
            cols["f_A"] = np.zeros(N, dtype=float)

        keys = list(g0.keys())
        for key in keys:
            cols[key] = np.zeros(N, dtype=float)

        n_filters = len(filter_names)
        _seds = np.zeros((N, n_filters), dtype=float)
        if absflux_cov:
            n_offdiag = ((n_filters**2) - n_filters) / 2
            _cov_diag = np.zeros((N, n_filters), dtype=float)
            _cov_offdiag = np.zeros((N, n_offdiag), dtype=float)

        with profile_stage("physics: dust SED generation"):
            for count, pt in enumerate(tqdm(chunk_pts, desc="SED grid")):

                if with_fA:
                    Av, Rv, f_A = pt
                    Rv_MW = extLaw.get_Rv_A(Rv, f_A)

                    r = g0.applyExtinctionLaw(
                        extLaw, Av=Av, Rv=Rv, f_A=f_A, inplace=False
                    )
                    # add extra "spectral bands" if requested
                    if add_spectral_properties_kwargs is not None:
                        r = add_spectral_properties(
                            r,
                            nameformat=nameformat,
                            filterLib=filterLib,
                            **add_spectral_properties_kwargs,
                        )
                    temp_results = r.getSEDs(filter_names, filterLib=filterLib)
                    # adding the dust parameters to the models
                    cols["Av"][N0 * count : N0 * (count + 1)] = Av
                    cols["Rv"][N0 * count : N0 * (count + 1)] = Rv
                    cols["f_A"][N0 * count : N0 * (count + 1)] = f_A
                    cols["Rv_A"][N0 * count : N0 * (count + 1)] = Rv_MW

                else:
                    Av, Rv = pt
                    r = g0.applyExtinctionLaw(extLaw, Av=Av, Rv=Rv, inplace=False)

                    if add_spectral_properties_kwargs is not None:
                        r = add_spectral_properties(
                            r,
                            nameformat=nameformat,
                            filterLib=filterLib,
                            **add_spectral_properties_kwargs,
                        )
                    temp_results = r.getSEDs(filter_names, filterLib=filterLib)
                    # adding the dust parameters to the models
                    cols["Av"][N0 * count : N0 * (count + 1)] = Av
                    cols["Rv"][N0 * count : N0 * (count + 1)] = Rv

                # compute the dust weights
                dust_prior_weight = compute_av_rv_fA_prior_weights(
                    Av,
                    Rv,
                    f_A,
                    g0.grid["distance"].data,
                    av_prior_model=av_prior_model,
                    rv_prior_model=rv_prior_model,
                    fA_prior_model=fA_prior_model,
                )

                # get new attributes if exist
                for key in list(temp_results.grid.keys()):
                    if key not in keys:
                        k1 = N0 * count
                        k2 = N0 * (count + 1)
                        cols.setdefault(key, np.zeros(N, dtype=float))[k1:k2] = (
                            temp_results.grid[key]
                        )

                # compute the fractional absflux covariance matrices
                if absflux_cov:
                    absflux_covmats = calc_absflux_cov_matrices(
                        r, temp_results, filter_names
                    )
                    _cov_diag[N0 * count : N0 * (count + 1)] = absflux_covmats[0]
                    _cov_offdiag[N0 * count : N0 * (count + 1)] = absflux_covmats[1]

                # assign the extinguished SEDs to the output object
                _seds[N0 * count : N0 * (count + 1)] = temp_results.seds[:]

                # copy the rest of the parameters
                for key in keys:
                    cols[key][N0 * count : N0 * (count + 1)] = g0.grid[key]

                # multiply existing prior weights by the dust prior weight
                cols["weight"][N0 * count : N0 * (count + 1)] *= dust_prior_weight
                cols["prior_weight"][
                    N0 * count : N0 * (count + 1)
                ] *= dust_prior_weight

                if count == 0:
                    cols["lamb"] = temp_results.lamb[:]

        _lamb = cols.pop("lamb")

        # now add the grid weights
        av_grid_weights = compute_grid_weights(avs)
        for cav, cav_gweight in zip(avs, av_grid_weights):
            gvals = cols["Av"] == cav
            cols["weight"][gvals] *= cav_gweight
            cols["grid_weight"][gvals] *= cav_gweight

        rv_grid_weights = compute_grid_weights(rvs)
        for rav, rav_gweight in zip(rvs, rv_grid_weights):
            gvals = cols["Rv"] == rav
            cols["weight"][gvals] *= rav_gweight
            cols["grid_weight"][gvals] *= rav_gweight

        fA_grid_weights = compute_grid_weights(fAs)
        for cfA, cfA_gweight in zip(fAs, fA_grid_weights):
            gvals = cols["f_A"] == cfA
            cols["weight"][gvals] *= cfA_gweight
            cols["grid_weight"][gvals] *= cfA_gweight

        # free the memory of temp_results
        # del temp_results
        # del tempgrid

        # Ship
        if absflux_cov:
            g = SEDGrid(
                _lamb,
                seds=_seds,
                cov_diag=_cov_diag,
                cov_offdiag=_cov_offdiag,
                grid=Table(cols),
                backend="memory",
            )
        else:
            g = SEDGrid(_lamb, seds=_seds, grid=Table(cols), backend="memory")

        g.header["filters"] = " ".join(filter_names)

        yield g


@generator
def make_extinguished_grid_fast(
    spec_grid,
    filter_names,
    extLaw,
    avs,
    rvs,
    fAs=None,
    av_prior_model={"name": "flat"},
    rv_prior_model={"name": "flat"},
    fA_prior_model={"name": "flat"},
    chunksize=0,
    add_spectral_properties_kwargs=None,
    absflux_cov=False,
    filterLib=None,
):
    """
    Vectorized equivalent of make_extinguished_grid for the common SED case.

    This path keeps the same dust-point ordering and table columns as the
    original implementation, but preloads filters and precomputes the trapezoid
    integration operator once. It intentionally does not implement absolute
    flux covariance matrices, where the existing object path is still the
    reference implementation.
    """
    if absflux_cov:
        raise NotImplementedError(
            "make_extinguished_grid_fast does not support absflux_cov"
        )

    add_props = None
    if add_spectral_properties_kwargs is not None:
        add_props = dict(add_spectral_properties_kwargs)
        if add_props.get("callables") is not None:
            raise NotImplementedError(
                "make_extinguished_grid_fast does not support callable spectral properties"
            )

    if isinstance(spec_grid, str):
        ext = spec_grid.split(".")[-1]
        if ext in ["hdf", "hd5", "hdf5"]:
            g0 = SpectralGrid(spec_grid, backend="disk")
        else:
            g0 = SpectralGrid(spec_grid, backend="cache")
    else:
        helpers.type_checker("spec_grid", spec_grid, SpectralGrid)
        g0 = spec_grid

    with_fA = fAs is not None
    min_Rv = min(rvs)
    max_Rv = max(rvs)

    if with_fA:
        Rv_B = extLaw.BLaw.Rv
        it = np.nditer(np.ix_(avs, rvs, fAs))
        niter = np.size(avs) * np.size(rvs) * np.size(fAs)
        npts, pts = _make_dust_fA_valid_points_generator(it, min_Rv, max_Rv, Rv_B)
        print(
            """number of initially requested points = {0:d}
              number of valid points = {1:d} (based on restrictions in R(V)
                 versus f_A plane)
              """.format(
                niter, npts
            )
        )
        if npts == 0:
            raise AttributeError("No valid points")
        dust_points = list(pts)
    else:
        dust_points = [(float(ak), float(rk)) for ak, rk in np.nditer(np.ix_(avs, rvs))]
        npts = len(dust_points)

    N0 = len(g0.grid)
    N = N0 * npts
    if chunksize <= 0:
        chunksize = npts
        print("Generating a final grid of {0:d} points".format(N))
    else:
        print(
            "Generating a final grid of {0:d} points in {1:d} pieces".format(
                N, int(float(npts) / chunksize + 1.0)
            )
        )

    if isinstance(filter_names[0], str):
        flist = phot.load_filters(
            filter_names, interp=True, lamb=g0.lamb, filterLib=filterLib
        )
        out_filter_names = filter_names
    else:
        flist = phot.load_Integrationfilters(filter_names, interp=True, lamb=g0.lamb)
        out_filter_names = [fk.name for fk in filter_names]

    _lamb, sed_matrix = _filter_integration_matrix(g0.lamb, flist, absFlux=True)
    prop_matrices = []
    if add_props is not None:
        nameformat = add_props.pop("nameformat", "{0:s}") + "_wd"
        prop_filternames = add_props.pop("filternames", None)
        prop_filters = add_props.pop("filters", None)
        # The original function forwards filterLib for named filters.
        if prop_filternames is not None:
            prop_flist = phot.load_filters(
                prop_filternames, interp=True, lamb=g0.lamb, filterLib=filterLib
            )
            _, prop_matrix = _filter_integration_matrix(
                g0.lamb, prop_flist, absFlux=True
            )
            prop_matrices.append((prop_filternames, prop_matrix, nameformat))
        if prop_filters is not None:
            prop_flist = phot.load_Integrationfilters(
                prop_filters, interp=True, lamb=g0.lamb
            )
            _, prop_matrix = _filter_integration_matrix(
                g0.lamb, prop_flist, absFlux=True
            )
            prop_matrices.append(
                ([fk.name for fk in prop_filters], prop_matrix, nameformat)
            )
        if add_props:
            raise NotImplementedError(
                "make_extinguished_grid_fast only supports filternames/filters spectral properties"
            )

    base_seds = np.asarray(g0.seds)
    keys = list(g0.keys())
    base_cols = {key: np.asarray(g0.grid[key]) for key in keys}
    base_distances = np.asarray(g0.grid["distance"].data)

    for chunk_points in helpers.chunks(dust_points, chunksize):
        chunk_points = list(chunk_points)
        chunk_npts = len(chunk_points)
        chunk_N = N0 * chunk_npts

        cols = {
            "Av": np.zeros(chunk_N, dtype=float),
            "Rv": np.zeros(chunk_N, dtype=float),
        }
        if with_fA:
            cols["Rv_A"] = np.zeros(chunk_N, dtype=float)
            cols["f_A"] = np.zeros(chunk_N, dtype=float)
        for key in keys:
            cols[key] = np.tile(base_cols[key], chunk_npts)

        _seds = np.zeros((chunk_N, len(flist)), dtype=float)
        prop_seds_by_name = {}

        with profile_stage("physics: dust SED generation"):
            for count, pt in enumerate(tqdm(chunk_points, desc="SED grid")):
                k1 = N0 * count
                k2 = N0 * (count + 1)
                f_A = 1.0
                if with_fA:
                    Av, Rv, f_A = pt
                    Rv_A = extLaw.get_Rv_A(Rv, f_A)
                    ext_curve = np.exp(
                        -1.0 * extLaw.function(g0.lamb[:], Av=Av, Rv=Rv, f_A=f_A)
                    )
                    cols["f_A"][k1:k2] = f_A
                    cols["Rv_A"][k1:k2] = Rv_A
                else:
                    Av, Rv = pt
                    ext_curve = np.exp(
                        -1.0 * extLaw.function(g0.lamb[:], Av=Av, Rv=Rv)
                    )

                attenuated_seds = base_seds * ext_curve[None, :]
                temp_seds = attenuated_seds @ sed_matrix
                _seds[k1:k2] = temp_seds
                cols["Av"][k1:k2] = Av
                cols["Rv"][k1:k2] = Rv

                for prop_names, prop_matrix, nameformat in prop_matrices:
                    prop_values = attenuated_seds @ prop_matrix
                    log_prop_values = _log_sed_columns(prop_values)
                    for i, fk in enumerate(prop_names):
                        colname = "log" + nameformat.format(fk)
                        prop_seds_by_name.setdefault(
                            colname, np.zeros(chunk_N, dtype=float)
                        )[k1:k2] = log_prop_values[:, i]

                dust_prior_weight = compute_av_rv_fA_prior_weights(
                    Av,
                    Rv,
                    f_A,
                    base_distances,
                    av_prior_model=av_prior_model,
                    rv_prior_model=rv_prior_model,
                    fA_prior_model=fA_prior_model,
                )
                cols["weight"][k1:k2] *= dust_prior_weight
                cols["prior_weight"][k1:k2] *= dust_prior_weight

        cols.update(prop_seds_by_name)

        av_grid_weights = compute_grid_weights(avs)
        for cav, cav_gweight in zip(avs, av_grid_weights):
            gvals = cols["Av"] == cav
            cols["weight"][gvals] *= cav_gweight
            cols["grid_weight"][gvals] *= cav_gweight

        rv_grid_weights = compute_grid_weights(rvs)
        for rav, rav_gweight in zip(rvs, rv_grid_weights):
            gvals = cols["Rv"] == rav
            cols["weight"][gvals] *= rav_gweight
            cols["grid_weight"][gvals] *= rav_gweight

        if with_fA:
            fA_grid_weights = compute_grid_weights(fAs)
            for cfA, cfA_gweight in zip(fAs, fA_grid_weights):
                gvals = cols["f_A"] == cfA
                cols["weight"][gvals] *= cfA_gweight
                cols["grid_weight"][gvals] *= cfA_gweight

        g = SEDGrid(_lamb, seds=_seds, grid=Table(cols), backend="memory")
        g.header["filters"] = " ".join(out_filter_names)
        yield g


def add_spectral_properties(
    specgrid,
    filternames=None,
    filters=None,
    callables=None,
    nameformat=None,
    filterLib=None,
):
    """
    Addon spectral calculations to spectral grids to extract in the fitting
    routines

    Parameters
    ----------
    specgrid: SpectralGrid instance
        instance of the spectral grid

    filternames: sequence(str)
        compute the integrated values through given filters in the library

    filters: sequence(Filters)
        sequence of filter instances from which extract integrated values

    callables: sequence(callable)
        sequence of functions to apply onto the spectral grid assuming storing
        results is internally processed by the individual functions

    nameformat: str
        naming format to adopt for filternames and filters
        default value is '{0:s}_0' where the value will be the filter name

    filterLib:  str
        full filename to the filter library hd5 file

    Returns
    -------
    specgrid: SpectralGrid instance
        instance of the input spectral grid which will include more properties
    """
    if nameformat is None:
        nameformat = "{0:s}_0"

    if filternames is not None:
        temp = specgrid.getSEDs(filternames, extLaw=None, filterLib=filterLib)

        logtempseds = np.array(temp.seds)
        indxs = np.where(temp.seds > 0)
        if len(indxs) > 0:
            logtempseds[indxs] = np.log10(temp.seds[indxs])
        indxs = np.where(temp.seds <= 0)
        if len(indxs) > 0:
            logtempseds[indxs] = -100.0

        for i, fk in enumerate(filternames):
            specgrid.grid["log" + nameformat.format(fk)] = logtempseds[:, i]
        del temp

    if filters is not None:
        temp = specgrid.getSEDs(filters, extLaw=None)

        logtempseds = np.array(temp.seds)
        indxs = np.where(temp.seds > 0)
        if len(indxs) > 0:
            logtempseds[indxs] = np.log10(temp.seds[indxs])

        indxs = np.where(temp.seds <= 0)
        if len(indxs) > 0:
            logtempseds[indxs] = -100.0

        for i, fk in enumerate(filters):
            specgrid.grid["log" + nameformat.format(fk.name)] = logtempseds[:, i]
        del temp

    if callables is not None:
        for fn in callables:
            fn(specgrid)

    return specgrid


def calc_absflux_cov_matrices(specgrid, sedgrid, filter_names):
    """Calculate the absflux covariance matrices for each model
    Must be done on the full spectrum of each model to account for
    the changing combined spectral response due to the model SED and
    the filter response curve.

    Parameters
    ----------
    specgrid: SpectralGrid instance
        instance of the spectral grid containing the full spectrum fluxes

    sedgrid: SpectralGrid instance
        instance of the spectral grid containing the band SED fluxes

    Returns
    -------
    absflux_covmat :
    """

    # get the fractional absflux covariance matrix
    absflux_cov_mats = absflux_covmat.hst_frac_matrix(
        filter_names, spectrum=(specgrid.lamb[:], specgrid.seds)
    )

    # setup the output quantities
    n_models = specgrid.seds.shape[0]
    n_filters = len(filter_names)
    n_offdiag = ((n_filters**2) - n_filters) / 2
    cov_diag = np.zeros((n_models, n_filters), dtype=np.float64)
    cov_offdiag = np.zeros((n_models, n_offdiag), dtype=np.float64)

    # pack the resulting covariance matrices into diganonal and
    # non-diagnonal terms
    #   much more efficient for use later in combining with AST results
    #     and fitting
    #   also convert from fractional to physical flux units
    m = 0
    cov_diag[:, n_filters - 1] = absflux_cov_mats[
        :, n_filters - 1, n_filters - 1
    ] * np.square(sedgrid.seds[:, n_filters - 1])
    for k in range(n_filters - 1):
        cov_diag[:, k] = absflux_cov_mats[:, k, k] * np.square(sedgrid.seds[:, k])
        for ll in range(k + 1, n_filters):
            cov_offdiag[:, m] = (
                absflux_cov_mats[:, k, ll] * sedgrid.seds[:, k] * sedgrid.seds[:, ll]
            )
            m += 1

    return (cov_diag, cov_offdiag)

"""
BEAST Fitting functions
"""

import numpy as np
import tables
import string
from itertools import islice
import warnings

import numexpr
from itertools import combinations

from astropy import units as ap_units
from astropy.coordinates import SkyCoord as ap_SkyCoord

from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm

from beast.physicsmodel import grid
from beast.tools.symlog import symlog
from beast.fitting.fit_metrics.likelihood import (
    N_covar_logLikelihood,
    N_logLikelihood_NM,
)
from beast.fitting.fit_metrics import expectation, percentile
from beast.fitting.pdf1d import pdf1d
from beast.fitting.pdf2d import pdf2d
from beast.tools.profiling import profile_stage, profile_summary

_JAX_DIAG_LOGLIKE = None

__all__ = [
    "summary_table_memory",
    "Q_all_memory",
    "q_all_memory_batched_kernel",
    "q_all_memory_batched_blocked_kernel",
    "IAU_names_and_extra_info",
    "save_stats",
    "save_pdf1d",
    "save_lnp",
]


def save_stats(
    stats_outname,
    stats_dict_in,
    best_vals,
    exp_vals,
    per_vals,
    chi2_vals,
    chi2_indx,
    lnp_vals,
    lnp_indx,
    best_specgrid_indx,
    total_log_norm,
    qnames,
    p,
    filters,
    wavelengths,
):
    """
    Save various fitting statistics to a file

    Parameters
    ----------
    stats_outname : str
        output filename
    stats_dict_in : dict
        input dictonary with ancilliary info
    best_vals : ndarray
        2D `float` array of the best fit parameters
    exp_vals : ndarray
        2D `float` array of the expectation fit parameters
    per_vals : ndarray
        3D `float` array of the percentile fit parameters
    chi2_vals : ndarray
        1D `float` array of the chisqr values (does not include model weights)
    chi2_indx : ndarray
        1D `float` array of the indx in model grid of chisqr values
    lnp_vals : ndarray
        1D `float` array of the P(max) values (includes model weights)
    lnp_indx : ndarray
        1D `int` array of the indx in model grid of P(max) values
    best_specgrid_indx : ndarray
        1D `int` array of the indx in spectroscopic model grid of P(max) values
    total_log_norm : ndarray
        1D `float` array of the log of the total grid weight
    qnames : list
        list of the parameter names
    p : list
        list of percentiles use to create the per_vals

    Returns
    -------
    N/A
    """

    with profile_stage("output writing", detail=stats_outname):
        return _save_stats_impl(
            stats_outname,
            stats_dict_in,
            best_vals,
            exp_vals,
            per_vals,
            chi2_vals,
            chi2_indx,
            lnp_vals,
            lnp_indx,
            best_specgrid_indx,
            total_log_norm,
            qnames,
            p,
            filters,
            wavelengths,
        )


def _save_stats_impl(
    stats_outname,
    stats_dict_in,
    best_vals,
    exp_vals,
    per_vals,
    chi2_vals,
    chi2_indx,
    lnp_vals,
    lnp_indx,
    best_specgrid_indx,
    total_log_norm,
    qnames,
    p,
    filters,
    wavelengths,
):
    stats_dict = stats_dict_in.copy()

    # populate the dict array
    for k, qname in enumerate(qnames):
        stats_dict["{0:s}_Best".format(qname)] = best_vals[:, k]
        stats_dict["{0:s}_Exp".format(qname)] = exp_vals[:, k]
        for i, pval in enumerate(p):
            stats_dict["{0:s}_p{1:d}".format(qname, int(pval))] = per_vals[:, k, i]

    stats_dict["chi2min"] = chi2_vals
    stats_dict["chi2min_indx"] = chi2_indx.astype(int)
    stats_dict["Pmax"] = lnp_vals
    stats_dict["Pmax_indx"] = lnp_indx.astype(int)
    stats_dict["specgrid_indx"] = best_specgrid_indx.astype(int)
    stats_dict["total_log_norm"] = total_log_norm

    summary_tab = Table(stats_dict)

    if stats_outname is not None:
        # standard Table writing of FITS files does not support multiple extensions
        # but reading does, so only have to do this when writing
        ohdu = fits.HDUList()
        ohdu.append(fits.table_to_hdu(summary_tab))

        # create a table with the filter names and wavelengths
        # useful for plotting the results
        filters_tab = Table()
        filters_tab["filternames"] = filters
        filters_tab["wavelengths"] = wavelengths
        ohdu.append(fits.table_to_hdu(filters_tab))

        ohdu.writeto(stats_outname, overwrite=True)


def save_pdf1d(pdf1d_outname, save_pdf1d_vals, qnames):
    """
    Save the 1D PDFs to a file

    Parameters
    ----------
    pdf1d_outname : str
        output filename
    save_pdf1d_vals : list
        list of 2D nparrays giving the 1D PDFs for each parameter/variable
    qnames : list
        list of the parameter names

    Returns
    -------
    N/A
    """

    with profile_stage("output writing", detail=pdf1d_outname):
        return _save_pdf1d_impl(pdf1d_outname, save_pdf1d_vals, qnames)


def _save_pdf1d_impl(pdf1d_outname, save_pdf1d_vals, qnames):
    # write a small primary header
    fits.writeto(pdf1d_outname, np.zeros((2, 2)), overwrite=True)

    # write the 1D PDFs for all the objects, 1 set per extension
    for k, qname in enumerate(qnames):
        hdu = fits.PrimaryHDU(save_pdf1d_vals[k])
        pheader = hdu.header
        pheader.set("XTENSION", "IMAGE")
        pheader.set("EXTNAME", qname)
        fits.append(pdf1d_outname, save_pdf1d_vals[k], header=pheader)


def save_pdf2d(pdf2d_outname, save_pdf2d_vals, qname_pairs):
    """
    Save the 2D PDFs to a file

    Parameters
    ----------
    pdf2d_outname : str
        output filename
    save_pdf2d_vals : list of np.array
        list of 3D nparrays giving the 2D PDFs for each pair of parameters
    qname_pairs : list
        list of `str` giving the parameter pairs

    Returns
    -------
    N/A
    """

    with profile_stage("output writing", detail=pdf2d_outname):
        return _save_pdf2d_impl(pdf2d_outname, save_pdf2d_vals, qname_pairs)


def _save_pdf2d_impl(pdf2d_outname, save_pdf2d_vals, qname_pairs):
    # write a small primary header
    fits.writeto(pdf2d_outname, np.zeros((2, 2)), overwrite=True)

    # write the 2D PDFs for all the objects, 1 set per extension
    for k, qname_pair in enumerate(qname_pairs):
        hdu = fits.PrimaryHDU(save_pdf2d_vals[k])
        pheader = hdu.header
        pheader.set("XTENSION", "IMAGE")
        pheader.set("EXTNAME", qname_pair)
        fits.append(pdf2d_outname, save_pdf2d_vals[k], header=pheader)


def save_lnp(lnp_outname, save_lnp_vals):
    """
    Save the nD lnps to a file

    Parameters
    ----------
    lnp_outname : str
        output filename
    save_lnp_vals : list
        list of 5 parameter lists giving the lnp/chisqr info for each star

    Returns
    -------
    N/A
    """

    with profile_stage("output writing", detail=lnp_outname):
        return _save_lnp_impl(lnp_outname, save_lnp_vals)


def _save_lnp_impl(lnp_outname, save_lnp_vals):
    # code needed if hdf5 is corrupted - usually due to job ending in the
    #    middle of the writing of the lnp file
    #  should be rare (not originally as the lnp file was open and
    #    written to continuously -
    #                  should be fixed with the new code where the lnp
    #                  is saved every n stars instead)
    try:
        outfile = tables.open_file(lnp_outname, "a")
    except Exception:
        print(
            "partial run lnp file is corrupted - saving new lnp values in "
            + string.replace(lnp_outname, "lnp", "lnp_partial")
        )
        outfile = tables.open_file(
            string.replace(lnp_outname, "lnp", "lnp_partial"), "a"
        )

    for lnp_val in save_lnp_vals:
        e = lnp_val[0]
        try:
            star_group = outfile.create_group("/", "star_%d" % e, title="star %d" % e)
        except tables.exceptions.NodeError:
            # print('lnp for star ' + str(e) + ' already in file')
            pass
        else:
            outfile.create_array(star_group, "input", lnp_val[4])
            outfile.create_array(star_group, "idx", lnp_val[1])
            outfile.create_array(star_group, "lnp", lnp_val[2])
            outfile.create_array(star_group, "chi2", lnp_val[3])
    outfile.close()


def setup_param_bins(qname, max_nbins, g0, full_model_flux, filters, grid_info_dict):
    """
    Set up the bin properties for the given parameter

    Parameters
    ----------
    qname : str
        name of the parameter
    max_nbins : int
        max number of bins to use for the PDF calculations
    g0 : SEDGrid object
        the SED grid
    full_model_flux : ndarray
        1D `float` array of the fluxes for the model grid
    filters : list
        list of `str` of the names of the filters in the SED grid
    grid_info_dict : dict
        the override for bin min/max/n_bin

    Returns
    -------
    qname_vals : np.array
        1D `float` array with either the fluxes or the grid values
        for the input qname
    nbins : int
        number of bins
    logspacing : bool
        whether the bins should be log-spaced
    minval, maxval : floats
        min/max value for the bins
    """

    if "_bias" in qname:
        fname = (qname.replace("_wd_bias", "")).replace("symlog", "")
        qname_vals = full_model_flux[:, filters.index(fname)]
    else:
        qname_vals = g0[qname]

    if grid_info_dict is not None and qname in grid_info_dict:
        # When processing a subgrid, we actually need the number of
        # unique values across all the subgrids to make the 1dpdfs
        # compatible
        n_uniq = grid_info_dict[qname]["num_unique"]
        uniqvals = grid_info_dict[qname]["unique_vals"]
    else:
        uniqvals = np.unique(qname_vals)
        n_uniq = len(uniqvals)

    if n_uniq > max_nbins:
        # limit the number of bins in the 1D likelihood for speed
        nbins = max_nbins
    else:
        nbins = n_uniq

    # temp code for BEAST paper figure
    # if qname == "Z":
    #     nbins = nbins + 1

    # setup for the fast 1D/2D PDFs

    # needed for mass parameters as they are stored as linear values
    # computationally, less bins needed if 1D PDFs done as log spacing
    if qname in set(["M_ini", "M_act", "radius"]):
        logspacing = True
    else:
        logspacing = False

    if grid_info_dict is not None and qname in grid_info_dict:
        minval = grid_info_dict[qname]["min"]
        maxval = grid_info_dict[qname]["max"]
    else:
        minval = None
        maxval = None

    return qname_vals, nbins, logspacing, minval, maxval, uniqvals


def _batch_hist1d(bin_idx, W, nbins):
    # dense: bin_idx (M,), W (B,M)
    B, M = W.shape
    rows = np.repeat(np.arange(B, dtype=np.int64), M)
    comp = rows * nbins + np.tile(bin_idx.astype(np.int64), B)
    out = np.bincount(comp, weights=W.ravel(), minlength=B * nbins)
    return out.reshape(B, nbins)


def _batch_hist1d_topk(bin_idx_2d, W, nbins):
    # topk: bin_idx_2d (B,K), W (B,K)
    B, K = W.shape
    rows = np.repeat(np.arange(B, dtype=np.int64), K)
    comp = rows * nbins + bin_idx_2d.astype(np.int64).ravel()
    out = np.bincount(comp, weights=W.ravel(), minlength=B * nbins)
    return out.reshape(B, nbins)


def _batch_hist2d(bin_idx2d, W, nbins2d):
    # dense: bin_idx2d (M,), W (B,M)
    B, M = W.shape
    rows = np.repeat(np.arange(B, dtype=np.int64), M)
    comp = rows * nbins2d + np.tile(bin_idx2d.astype(np.int64), B)
    out = np.bincount(comp, weights=W.ravel(), minlength=B * nbins2d)
    return out.reshape(B, nbins2d)


def _batch_hist2d_topk(bin_idx2d_2d, W, nbins2d):
    # topk: bin_idx2d_2d (B,K), W (B,K)
    B, K = W.shape
    rows = np.repeat(np.arange(B, dtype=np.int64), K)
    comp = rows * nbins2d + bin_idx2d_2d.astype(np.int64).ravel()
    out = np.bincount(comp, weights=W.ravel(), minlength=B * nbins2d)
    return out.reshape(B, nbins2d)


def _cdf_quantiles_from_pdf(pdf_vals, bin_vals, pcts):
    p = np.asarray(pcts, dtype=np.float64) / 100.0
    B, nbins = pdf_vals.shape
    out = np.zeros((B, len(p)), dtype=np.float64)

    # Match beast.fitting.fit_metrics.common.percentile exactly: percentiles
    # are interpolated at the center of each bin's weight, not at CDF edges.
    cdf = np.cumsum(pdf_vals, axis=1)
    totals = cdf[:, -1]
    valid = totals > 0.0
    if not np.any(valid):
        return out

    wpos = (cdf[valid] - 0.5 * pdf_vals[valid]) / totals[valid, None]
    for i in range(wpos.shape[0]):
        out[np.flatnonzero(valid)[i], :] = np.interp(p, wpos[i], bin_vals)

    return out

def _as_model_filter(a, n_filters):
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array, got {a.shape}")
    if a.shape[1] == n_filters:
        return a
    if a.shape[0] == n_filters:
        return a.T
    raise ValueError(f"Cannot infer (M,F) layout from shape {a.shape} and n_filters={n_filters}")

# ============================================================
# Top-K selection from retained mass + ESS(K)
# ============================================================

def _choose_topk_from_weights(
    W,
    mass_target=0.999,
    ess_target=128.0,
    k_min=32,
    k_max=2048,
    quantile=0.95,
    return_diagnostics=False,
):
    """
    W: (B, M) normalized weights

    ESS(K) = (sum_{i<=K} w_i)^2 / sum_{i<=K} w_i^2, with weights sorted descending.

    Returns
    -------
    order   : (B,M) descending sort indices
    K_star  : (B,) per-star chosen K
    K_batch : int batch-wide K used for vectorization
    and optionally W_sorted, rho_curve, ess_curve
    """
    B, M = W.shape

    order = np.argsort(-W, axis=1)
    W_sorted = np.take_along_axis(W, order, axis=1)

    rho_curve = np.cumsum(W_sorted, axis=1)
    s2_curve = np.cumsum(W_sorted * W_sorted, axis=1)
    ess_curve = np.divide(rho_curve * rho_curve, s2_curve, out=np.zeros_like(rho_curve), where=s2_curve > 0)

    Kgrid = np.arange(1, M + 1)[None, :]
    ok = (rho_curve >= mass_target) & (ess_curve >= ess_target) & (Kgrid >= k_min)

    if k_max is not None:
        ok &= (Kgrid <= min(k_max, M))

    has_ok = ok.any(axis=1)
    fallback = min(k_max if k_max is not None else M, M)
    K_star = np.where(has_ok, ok.argmax(axis=1) + 1, fallback)

    K_batch = int(np.quantile(K_star, quantile))
    K_batch = max(k_min, K_batch)
    if k_max is not None:
        K_batch = min(K_batch, k_max)
    K_batch = min(K_batch, M)

    if return_diagnostics:
        return order, K_star, K_batch, W_sorted, rho_curve, ess_curve
    return order, K_star, K_batch


def _batched_diag_loglike(Y, mu, ivar):
    """
    Batched version of N_logLikelihood_NM for the no-mask case.

    Parameters
    ----------
    Y : (B, F)
        observed SEDs
    mu : (M, F)
        model_seds_with_bias
    ivar : (M, F)
        ast_ivar

    Returns
    -------
    lnP : (B, M)
    chi2 : (B, M)
    """
    Y = np.asarray(Y, dtype=np.float64, order="C")
    mu = np.asarray(mu, dtype=np.float64, order="C")
    ivar = np.asarray(ivar, dtype=np.float64, order="C")

    temp = 0.5 * np.log(2.0 * np.pi)
    n = Y.shape[1]
    lnQ = n * temp - 0.5 * np.sum(np.log(ivar), axis=1)  # (M,)

    A = ivar * mu
    c = np.einsum("mf,mf->m", A, mu, optimize=True)

    chi2 = (Y * Y) @ ivar.T
    chi2 -= 2.0 * (Y @ A.T)
    chi2 += c[None, :]

    lnP = -lnQ[None, :] - 0.5 * chi2
    return lnP, chi2


def _batched_diag_loglike_jax(Y, mu, ivar):
    """JAX implementation of the diagonal batched likelihood."""
    global _JAX_DIAG_LOGLIKE
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "backend='jax' requires jax to be installed; use backend='numpy' "
            "or install jax separately."
        ) from exc

    jax.config.update("jax_enable_x64", True)

    if _JAX_DIAG_LOGLIKE is None:

        @jax.jit
        def _kernel(Y_j, mu_j, ivar_j):
            temp = 0.5 * jnp.log(2.0 * jnp.pi)
            n = Y_j.shape[1]
            lnQ = n * temp - 0.5 * jnp.sum(jnp.log(ivar_j), axis=1)

            A = ivar_j * mu_j
            c = jnp.einsum("mf,mf->m", A, mu_j, optimize=True)

            chi2 = (Y_j * Y_j) @ ivar_j.T
            chi2 = chi2 - 2.0 * (Y_j @ A.T)
            chi2 = chi2 + c[None, :]

            lnP = -lnQ[None, :] - 0.5 * chi2
            return lnP, chi2

        _JAX_DIAG_LOGLIKE = _kernel

    lnp, chi2 = _JAX_DIAG_LOGLIKE(
        jnp.asarray(Y, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(ivar, dtype=jnp.float64),
    )
    return np.asarray(lnp), np.asarray(chi2)


def _batched_fullcov_loglike(Y, mu, q_norm, icov_diag, two_icov_offdiag):
    """
    Batched version of N_covar_logLikelihood.

    Parameters
    ----------
    Y : (B, F)
        observed SEDs
    mu : (M, F)
        model_seds_with_bias
    q_norm : (M,)
        ast_q_norm
    icov_diag : (M, F)
        ast_icov_diag
    two_icov_offdiag : (M, K)
        two_ast_icov_offdiag, with K = F*(F-1)/2

    Returns
    -------
    lnP : (B, M)
    chi2 : (B, M)
    """
    Y = np.asarray(Y, dtype=np.float64, order="C")
    mu = np.asarray(mu, dtype=np.float64, order="C")
    q_norm = np.asarray(q_norm, dtype=np.float64, order="C")
    icov_diag = np.asarray(icov_diag, dtype=np.float64, order="C")
    two_icov_offdiag = np.asarray(two_icov_offdiag, dtype=np.float64, order="C")

    M, F = icov_diag.shape
    A = np.zeros((M, F, F), dtype=np.float64)

    for i in range(F):
        A[:, i, i] = icov_diag[:, i]

    k = 0
    for i in range(F):
        for j in range(i + 1, F):
            off = 0.5 * two_icov_offdiag[:, k]
            A[:, i, j] = off
            A[:, j, i] = off
            k += 1

    b = np.einsum("mfg,mg->mf", A, mu, optimize=True)
    c = np.einsum("mf,mf->m", mu, b, optimize=True)

    quad = np.einsum("bf,mfg,bg->bm", Y, A, Y, optimize=True)
    cross = np.einsum("bf,mf->bm", Y, b, optimize=True)

    chi2 = quad - 2.0 * cross + c[None, :]
    lnP = q_norm[None, :] - 0.5 * chi2
    return lnP, chi2


def q_all_memory_batched_kernel(
    Y_batch,
    model_seds_with_bias,
    log_prior_weights,
    q_param_arrays,
    pdf1d_bin_indices,
    pdf1d_bin_values,
    pdf2d_flat_indices=(),
    pdf2d_shapes=(),
    ast_ivar=None,
    ast_q_norm=None,
    ast_icov_diag=None,
    two_ast_icov_offdiag=None,
    use_full_cov_matrix=False,
    threshold=-40.0,
    p=(16.0, 50.0, 84.0),
    compute_percentiles=True,
    fit_use_topk=False,
    fit_topk_mass_target=0.999,
    fit_topk_ess_target=128.0,
    fit_topk_kmin=32,
    fit_topk_kmax=2048,
    fit_topk_kquantile=0.95,
    backend="numpy",
):
    """Pure array kernel for one batched fitting block.

    All inputs are arrays/scalars and all returned values are arrays or lists
    of arrays. File I/O, table handling, progress reporting, and grid object
    access stay in the caller.
    """
    if backend not in {"numpy", "jax"}:
        raise ValueError("backend must be either 'numpy' or 'jax'")

    if use_full_cov_matrix:
        if backend == "jax":
            raise NotImplementedError(
                "backend='jax' is only implemented for diagonal likelihoods"
            )
        lnp0, chi2 = _batched_fullcov_loglike(
            Y_batch,
            model_seds_with_bias,
            ast_q_norm,
            ast_icov_diag,
            two_ast_icov_offdiag,
        )
    elif backend == "jax":
        lnp0, chi2 = _batched_diag_loglike_jax(
            Y_batch,
            model_seds_with_bias,
            ast_ivar,
        )
    else:
        lnp0, chi2 = _batched_diag_loglike(
            Y_batch,
            model_seds_with_bias,
            ast_ivar,
        )

    lnp = lnp0 + log_prior_weights[None, :]
    B = Y_batch.shape[0]

    max_lnp = np.max(np.where(np.isfinite(lnp), lnp, -np.inf), axis=1)
    keep = (lnp - max_lnp[:, None]) > threshold

    logw = lnp - max_lnp[:, None]
    logw = np.where(keep, logw, -np.inf)
    w = np.exp(np.clip(logw, -700.0, 0.0))
    sumw = np.sum(w, axis=1, keepdims=True)
    posterior_weights = np.divide(w, sumw, out=np.zeros_like(w), where=sumw > 0)

    lnp_local_indices = np.argmax(lnp, axis=1)
    chi2_local_indices = np.argmin(chi2, axis=1)
    total_log_norm = max_lnp + np.log(sumw[:, 0])

    if fit_use_topk:
        order, K_star, K_batch, _, _, _ = _choose_topk_from_weights(
            posterior_weights,
            mass_target=fit_topk_mass_target,
            ess_target=fit_topk_ess_target,
            k_min=fit_topk_kmin,
            k_max=fit_topk_kmax,
            quantile=fit_topk_kquantile,
            return_diagnostics=True,
        )
        weight_indices = order[:, :K_batch]
        weights_used = np.take_along_axis(posterior_weights, weight_indices, axis=1)
        weights_used /= np.maximum(
            weights_used.sum(axis=1, keepdims=True), np.finfo(np.float64).tiny
        )
    else:
        weight_indices = None
        weights_used = posterior_weights

    q_param_arrays = np.asarray(q_param_arrays, dtype=np.float64)
    nq = q_param_arrays.shape[0]
    best_vals = q_param_arrays[:, lnp_local_indices].T
    exp_vals = np.zeros((B, nq), dtype=np.float64)

    for k in range(nq):
        q_act = q_param_arrays[k]
        if fit_use_topk:
            exp_vals[:, k] = np.sum(weights_used * q_act[weight_indices], axis=1)
        else:
            exp_vals[:, k] = weights_used @ q_act

    p = tuple(p) if compute_percentiles else ()

    pdf1d_batches = []
    per_vals = np.zeros((B, nq, len(p)), dtype=np.float64)
    for k, bin_idx in enumerate(pdf1d_bin_indices):
        bin_vals = pdf1d_bin_values[k]
        nb = len(bin_vals)
        if fit_use_topk:
            pdf_batch = _batch_hist1d_topk(bin_idx[weight_indices], weights_used, nb)
        else:
            pdf_batch = _batch_hist1d(bin_idx, weights_used, nb)

        pdf1d_batches.append(pdf_batch)
        if compute_percentiles:
            per_vals[:, k, :] = _cdf_quantiles_from_pdf(pdf_batch, bin_vals, p)

    pdf2d_batches = []
    for flat_idx, shape in zip(pdf2d_flat_indices, pdf2d_shapes):
        nb1, nb2 = shape
        if fit_use_topk:
            hist_flat = _batch_hist2d_topk(
                flat_idx[weight_indices], weights_used, nb1 * nb2
            )
        else:
            hist_flat = _batch_hist2d(flat_idx, weights_used, nb1 * nb2)
        pdf2d_batches.append(hist_flat.reshape(B, nb1, nb2))

    if fit_use_topk:
        retained_local_indices = [weight_indices[i].copy() for i in range(B)]
    else:
        retained_local_indices = [np.where(keep[i])[0] for i in range(B)]

    return {
        "lnp": lnp,
        "chi2": chi2,
        "keep": keep,
        "posterior_weights": posterior_weights,
        "weights_used": weights_used,
        "weight_indices": weight_indices,
        "retained_local_indices": retained_local_indices,
        "chi2_values": chi2[np.arange(B), chi2_local_indices],
        "chi2_local_indices": chi2_local_indices,
        "lnp_values": max_lnp,
        "lnp_local_indices": lnp_local_indices,
        "total_log_norm": total_log_norm,
        "best_vals": best_vals,
        "exp_vals": exp_vals,
        "per_vals": per_vals,
        "pdf1d_batches": pdf1d_batches,
        "pdf2d_batches": pdf2d_batches,
    }


def q_all_memory_batched_blocked_kernel(
    Y_batch,
    model_seds_with_bias,
    log_prior_weights,
    q_param_arrays,
    pdf1d_bin_indices,
    pdf1d_bin_values,
    pdf2d_flat_indices=(),
    pdf2d_shapes=(),
    ast_ivar=None,
    threshold=-40.0,
    p=(16.0, 50.0, 84.0),
    compute_percentiles=True,
    model_block_size=250000,
    backend="numpy",
    retain_sparse=True,
):
    """Memory-bounded diagonal batched fitting kernel.

    This is an exact dense-posterior kernel for the diagonal-noise case, but it
    streams over model blocks instead of materializing full ``B x M`` posterior
    arrays. It performs two passes over the model grid: the first pass finds the
    per-star posterior maximum and chi-square minimum, and the second pass
    accumulates thresholded posterior mass, expectations, PDFs, and optional
    sparse retained values.

    Full covariance likelihoods and approximate top-k summaries intentionally
    stay out of this kernel for now.
    """
    if backend not in {"numpy", "jax"}:
        raise ValueError("backend must be either 'numpy' or 'jax'")
    if model_block_size <= 0:
        raise ValueError("model_block_size must be positive")
    if ast_ivar is None:
        raise ValueError("ast_ivar is required for the diagonal blocked kernel")

    Y_batch = np.asarray(Y_batch, dtype=np.float64, order="C")
    model_seds_with_bias = np.asarray(
        model_seds_with_bias, dtype=np.float64, order="C"
    )
    ast_ivar = np.asarray(ast_ivar, dtype=np.float64, order="C")
    log_prior_weights = np.asarray(log_prior_weights, dtype=np.float64)
    q_param_arrays = np.asarray(q_param_arrays, dtype=np.float64)

    B = Y_batch.shape[0]
    M = model_seds_with_bias.shape[0]
    nq = q_param_arrays.shape[0]
    block_size = min(int(model_block_size), M)

    max_lnp = np.full(B, -np.inf, dtype=np.float64)
    lnp_local_indices = np.zeros(B, dtype=np.int64)
    min_chi2 = np.full(B, np.inf, dtype=np.float64)
    chi2_local_indices = np.zeros(B, dtype=np.int64)

    for m0 in range(0, M, block_size):
        m1 = min(M, m0 + block_size)
        lnp0, chi2 = (
            _batched_diag_loglike_jax(
                Y_batch, model_seds_with_bias[m0:m1], ast_ivar[m0:m1]
            )
            if backend == "jax"
            else _batched_diag_loglike(
                Y_batch, model_seds_with_bias[m0:m1], ast_ivar[m0:m1]
            )
        )
        lnp = lnp0 + log_prior_weights[m0:m1][None, :]

        block_lnp_idx = np.argmax(lnp, axis=1)
        block_lnp = lnp[np.arange(B), block_lnp_idx]
        update_lnp = block_lnp > max_lnp
        max_lnp[update_lnp] = block_lnp[update_lnp]
        lnp_local_indices[update_lnp] = m0 + block_lnp_idx[update_lnp]

        block_chi2_idx = np.argmin(chi2, axis=1)
        block_chi2 = chi2[np.arange(B), block_chi2_idx]
        update_chi2 = block_chi2 < min_chi2
        min_chi2[update_chi2] = block_chi2[update_chi2]
        chi2_local_indices[update_chi2] = m0 + block_chi2_idx[update_chi2]

    sumw = np.zeros(B, dtype=np.float64)
    exp_sums = np.zeros((B, nq), dtype=np.float64)
    p = tuple(p) if compute_percentiles else ()
    per_vals = np.zeros((B, nq, len(p)), dtype=np.float64)

    pdf1d_batches = [
        np.zeros((B, len(bin_vals)), dtype=np.float64)
        for bin_vals in pdf1d_bin_values
    ]
    pdf2d_batches = [
        np.zeros((B, nb1, nb2), dtype=np.float64) for nb1, nb2 in pdf2d_shapes
    ]

    retained_local_indices = [[] for _ in range(B)]
    retained_lnp = [[] for _ in range(B)]
    retained_chi2 = [[] for _ in range(B)]

    for m0 in range(0, M, block_size):
        m1 = min(M, m0 + block_size)
        lnp0, chi2 = (
            _batched_diag_loglike_jax(
                Y_batch, model_seds_with_bias[m0:m1], ast_ivar[m0:m1]
            )
            if backend == "jax"
            else _batched_diag_loglike(
                Y_batch, model_seds_with_bias[m0:m1], ast_ivar[m0:m1]
            )
        )
        lnp = lnp0 + log_prior_weights[m0:m1][None, :]
        keep = (lnp - max_lnp[:, None]) > threshold
        logw = np.where(keep, lnp - max_lnp[:, None], -np.inf)
        w = np.exp(np.clip(logw, -700.0, 0.0))
        sumw += np.sum(w, axis=1)

        for k in range(nq):
            exp_sums[:, k] += w @ q_param_arrays[k, m0:m1]

        for k, bin_idx in enumerate(pdf1d_bin_indices):
            pdf1d_batches[k] += _batch_hist1d(bin_idx[m0:m1], w, len(pdf1d_bin_values[k]))

        for k, (flat_idx, shape) in enumerate(zip(pdf2d_flat_indices, pdf2d_shapes)):
            nb1, nb2 = shape
            hist_flat = _batch_hist2d(flat_idx[m0:m1], w, nb1 * nb2)
            pdf2d_batches[k] += hist_flat.reshape(B, nb1, nb2)

        if retain_sparse:
            for bi in range(B):
                idx = np.flatnonzero(keep[bi])
                if idx.size == 0:
                    continue
                retained_local_indices[bi].append(m0 + idx)
                retained_lnp[bi].append(lnp[bi, idx])
                retained_chi2[bi].append(chi2[bi, idx])

    safe_sumw = np.maximum(sumw, np.finfo(np.float64).tiny)
    exp_vals = exp_sums / safe_sumw[:, None]
    for k in range(len(pdf1d_batches)):
        pdf1d_batches[k] /= safe_sumw[:, None]
    for k in range(len(pdf2d_batches)):
        pdf2d_batches[k] /= safe_sumw[:, None, None]

    for k, bin_vals in enumerate(pdf1d_bin_values):
        if compute_percentiles:
            per_vals[:, k, :] = _cdf_quantiles_from_pdf(pdf1d_batches[k], bin_vals, p)

    best_vals = q_param_arrays[:, lnp_local_indices].T
    total_log_norm = max_lnp + np.log(safe_sumw)

    if retain_sparse:
        retained_local_indices = [
            np.concatenate(chunks).astype(np.int64) if chunks else np.array([], dtype=np.int64)
            for chunks in retained_local_indices
        ]
        retained_lnp = [
            np.concatenate(chunks).astype(np.float64) if chunks else np.array([], dtype=np.float64)
            for chunks in retained_lnp
        ]
        retained_chi2 = [
            np.concatenate(chunks).astype(np.float64) if chunks else np.array([], dtype=np.float64)
            for chunks in retained_chi2
        ]
    else:
        retained_local_indices = [np.array([], dtype=np.int64) for _ in range(B)]
        retained_lnp = [np.array([], dtype=np.float64) for _ in range(B)]
        retained_chi2 = [np.array([], dtype=np.float64) for _ in range(B)]

    return {
        "lnp": None,
        "chi2": None,
        "keep": None,
        "posterior_weights": None,
        "weights_used": None,
        "weight_indices": None,
        "retained_local_indices": retained_local_indices,
        "retained_lnp": retained_lnp,
        "retained_chi2": retained_chi2,
        "chi2_values": min_chi2,
        "chi2_local_indices": chi2_local_indices,
        "lnp_values": max_lnp,
        "lnp_local_indices": lnp_local_indices,
        "total_log_norm": total_log_norm,
        "best_vals": best_vals,
        "exp_vals": exp_vals,
        "per_vals": per_vals,
        "pdf1d_batches": pdf1d_batches,
        "pdf2d_batches": pdf2d_batches,
    }


def Q_all_memory_batched(
    prev_result,
    obs,
    sedgrid,
    obsmodel,
    qnames_in,
    p=(16.0, 50.0, 84.0),
    resume=False,
    threshold=-40.0,
    save_every_npts=None,
    lnp_npts=None,
    max_nbins=200,
    stats_outname=None,
    pdf1d_outname=None,
    pdf2d_outname=None,
    pdf2d_param_list=None,
    grid_info_dict=None,
    lnp_outname=None,
    use_full_cov_matrix=False,
    do_not_normalize=False,
    fit_use_topk=False,
    fit_star_batch_size=512,
    fit_topk_mass_target=0.999,
    fit_topk_ess_target=128.0,
    fit_topk_kmin=32,
    fit_topk_kmax=2048,
    fit_topk_kquantile=0.95,
    fit_model_block_size=0,
    backend="numpy",
    compute_percentiles=True,
):
    if resume:
        raise NotImplementedError("resume=True not implemented in Q_all_memory_batched.")

    filters = obs.getFilters()
    Y_all = np.vstack([obj for _, obj in obs.enumobs()])   # (N, F)
    n_obs, n_filters = Y_all.shape

    model_seds = _as_model_filter(sedgrid.seds, n_filters)
    bias = _as_model_filter(obsmodel["bias"], n_filters)
    model_seds_with_bias = model_seds + bias

    g0_w = np.asarray(sedgrid["weight"])
    g0_indxs = np.where(g0_w > 0.0)[0]
    g0_weights = np.log(g0_w[g0_indxs])
    if not do_not_normalize:
        g0_weights -= np.max(g0_weights)

    mu = model_seds_with_bias[g0_indxs]
    qnames = list(qnames_in)
    nq = len(qnames)

    full_model_flux = np.asarray(sedgrid.seds)
    g0_specgrid_indx = np.asarray(sedgrid["specgrid_indx"])

    full_cov_mat = False
    if (
        use_full_cov_matrix
        and ("q_norm" in obsmodel.keys())
        and ("icov_diag" in obsmodel.keys())
        and ("icov_offdiag" in obsmodel.keys())
    ):
        full_cov_mat = True
        ast_q_norm = np.asarray(obsmodel["q_norm"])[g0_indxs]
        ast_icov_diag = _as_model_filter(obsmodel["icov_diag"], n_filters)[g0_indxs]
        two_ast_icov_offdiag = 2.0 * np.asarray(obsmodel["icov_offdiag"])[g0_indxs]
    else:
        ast_error = _as_model_filter(obsmodel["error"], n_filters)[g0_indxs]
        ast_ivar = 1.0 / np.maximum(ast_error, np.finfo(np.float64).tiny) ** 2

    if full_cov_mat:
        print("using full covariance matrix")
    else:
        print("not using full covariance matrix")

    best_vals = np.zeros((n_obs, nq), dtype=np.float64)
    exp_vals = np.zeros((n_obs, nq), dtype=np.float64)
    p = tuple(p) if compute_percentiles else ()
    per_vals = np.zeros((n_obs, nq, len(p)), dtype=np.float64)

    chi2_vals = np.zeros(n_obs, dtype=np.float64)
    chi2_indx = np.zeros(n_obs, dtype=np.int64)
    lnp_vals = np.zeros(n_obs, dtype=np.float64)
    lnp_indx = np.zeros(n_obs, dtype=np.int64)
    best_specgrid_indx = np.zeros(n_obs, dtype=np.int64)
    total_log_norm = np.zeros(n_obs, dtype=np.float64)

    save_lnp_vals = []

    q_arrays_full = []
    q_arrays_active = []
    for qname in qnames:
        if "_bias" in qname:
            fname = (qname.replace("_wd_bias", "")).replace("symlog", "")
            q_full = np.asarray(full_model_flux[:, filters.index(fname)])
        else:
            q_full = np.asarray(sedgrid[qname])
        q_arrays_full.append(q_full)
        q_arrays_active.append(q_full[g0_indxs])

    need_pdf1d = (pdf1d_outname is not None) or compute_percentiles

    pdf1d_infos = []
    save_pdf1d_vals = []
    if need_pdf1d:
        for qname in qnames:
            qvals, nbins, logspacing, minval, maxval, uniqvals = setup_param_bins(
                qname, max_nbins, sedgrid, full_model_flux, filters, grid_info_dict
            )

            if uniqvals is not None:
                bin_vals = np.asarray(uniqvals, dtype=np.float64)
            else:
                bin_vals = (
                    np.logspace(np.log10(minval), np.log10(maxval), nbins)
                    if logspacing else
                    np.linspace(minval, maxval, nbins)
                )

            q_active = q_arrays_active[qnames.index(qname)]
            idx = np.searchsorted(bin_vals, q_active, side="left")
            idx = np.clip(idx, 0, len(bin_vals) - 1)

            pdf1d_infos.append((bin_vals, idx, len(bin_vals)))
            if pdf1d_outname is not None:
                arr = np.zeros((n_obs + 2, len(bin_vals)), dtype=np.float32)
                arr[-1, :] = bin_vals.astype(np.float32)
                save_pdf1d_vals.append(arr)

    if pdf2d_param_list is None:
        pdf2d_qname_pairs = []
    else:
        pdf2d_qname_pairs = [f"{a}+{b}" for a, b in combinations(pdf2d_param_list, 2)]

    pdf2d_infos = []
    save_pdf2d_vals = []
    for pair in pdf2d_qname_pairs:
        q1name, q2name = pair.split("+")

        q1vals, nb1, lg1, mn1, mx1, uq1 = setup_param_bins(
            q1name, max_nbins, sedgrid, full_model_flux, filters, grid_info_dict
        )
        q2vals, nb2, lg2, mn2, mx2, uq2 = setup_param_bins(
            q2name, max_nbins, sedgrid, full_model_flux, filters, grid_info_dict
        )

        bin1 = np.asarray(uq1, dtype=np.float64) if uq1 is not None else (
            np.logspace(np.log10(mn1), np.log10(mx1), nb1) if lg1 else np.linspace(mn1, mx1, nb1)
        )
        bin2 = np.asarray(uq2, dtype=np.float64) if uq2 is not None else (
            np.logspace(np.log10(mn2), np.log10(mx2), nb2) if lg2 else np.linspace(mn2, mx2, nb2)
        )

        q1_full = q_arrays_full[qnames.index(q1name)] if q1name in qnames else np.asarray(sedgrid[q1name])
        q2_full = q_arrays_full[qnames.index(q2name)] if q2name in qnames else np.asarray(sedgrid[q2name])

        q1_active = q1_full[g0_indxs]
        q2_active = q2_full[g0_indxs]

        idx1 = np.searchsorted(bin1, q1_active, side="left")
        idx1 = np.clip(idx1, 0, len(bin1) - 1)
        idx2 = np.searchsorted(bin2, q2_active, side="left")
        idx2 = np.clip(idx2, 0, len(bin2) - 1)

        flat_idx = idx1 * len(bin2) + idx2
        pdf2d_infos.append((bin1, bin2, flat_idx, len(bin1), len(bin2)))

        arr = np.zeros((n_obs + 2, len(bin1), len(bin2)), dtype=np.float32)
        arr[-2, :, :] = np.tile(bin1[:, None], (1, len(bin2))).astype(np.float32)
        arr[-1, :, :] = np.tile(bin2[None, :], (len(bin1), 1)).astype(np.float32)
        save_pdf2d_vals.append(arr)

    q_param_arrays = np.vstack(q_arrays_active)
    pdf1d_bin_values = [info[0] for info in pdf1d_infos]
    pdf1d_bin_indices = [info[1] for info in pdf1d_infos]
    pdf2d_flat_indices = [info[2] for info in pdf2d_infos]
    pdf2d_shapes = [(info[3], info[4]) for info in pdf2d_infos]

    it = range(0, n_obs, fit_star_batch_size)
    for b0 in tqdm(it, total=int(np.ceil(n_obs / fit_star_batch_size)), desc="Batched Lnp/Stats"):
        b1 = min(n_obs, b0 + fit_star_batch_size)
        Y = Y_all[b0:b1]
        B = Y.shape[0]

        with profile_stage("likelihood computation", detail="batched kernel"):
            if fit_model_block_size:
                if full_cov_mat:
                    raise NotImplementedError(
                        "fit_model_block_size is currently implemented only for "
                        "diagonal likelihoods"
                    )
                if fit_use_topk:
                    raise NotImplementedError(
                        "fit_model_block_size does not yet support approximate top-k"
                    )
                kernel_result = q_all_memory_batched_blocked_kernel(
                    Y,
                    mu,
                    g0_weights,
                    q_param_arrays,
                    pdf1d_bin_indices,
                    pdf1d_bin_values,
                    pdf2d_flat_indices=pdf2d_flat_indices,
                    pdf2d_shapes=pdf2d_shapes,
                    ast_ivar=ast_ivar,
                    threshold=threshold,
                    p=p,
                    compute_percentiles=compute_percentiles,
                    model_block_size=fit_model_block_size,
                    backend=backend,
                )
            else:
                kernel_result = q_all_memory_batched_kernel(
                    Y,
                    mu,
                    g0_weights,
                    q_param_arrays,
                    pdf1d_bin_indices,
                    pdf1d_bin_values,
                    pdf2d_flat_indices=pdf2d_flat_indices,
                    pdf2d_shapes=pdf2d_shapes,
                    ast_ivar=None if full_cov_mat else ast_ivar,
                    ast_q_norm=ast_q_norm if full_cov_mat else None,
                    ast_icov_diag=ast_icov_diag if full_cov_mat else None,
                    two_ast_icov_offdiag=two_ast_icov_offdiag if full_cov_mat else None,
                    use_full_cov_matrix=full_cov_mat,
                    threshold=threshold,
                    p=p,
                    compute_percentiles=compute_percentiles,
                    fit_use_topk=fit_use_topk,
                    fit_topk_mass_target=fit_topk_mass_target,
                    fit_topk_ess_target=fit_topk_ess_target,
                    fit_topk_kmin=fit_topk_kmin,
                    fit_topk_kmax=fit_topk_kmax,
                    fit_topk_kquantile=fit_topk_kquantile,
                    backend=backend,
                )

        best_local = kernel_result["lnp_local_indices"]
        best_full = g0_indxs[best_local]
        chi2_local = kernel_result["chi2_local_indices"]
        chi2_full = g0_indxs[chi2_local]

        total_log_norm[b0:b1] = kernel_result["total_log_norm"]
        best_specgrid_indx[b0:b1] = g0_specgrid_indx[best_full]
        chi2_vals[b0:b1] = kernel_result["chi2_values"]
        chi2_indx[b0:b1] = chi2_full
        lnp_vals[b0:b1] = kernel_result["lnp_values"]
        lnp_indx[b0:b1] = best_full

        best_vals[b0:b1, :] = kernel_result["best_vals"]
        exp_vals[b0:b1, :] = kernel_result["exp_vals"]
        per_vals[b0:b1, :, :] = kernel_result["per_vals"]

        if pdf1d_outname is not None:
            for k, pdf_batch in enumerate(kernel_result["pdf1d_batches"]):
                save_pdf1d_vals[k][b0:b1, :] = pdf_batch.astype(np.float32)

        for k, hist in enumerate(kernel_result["pdf2d_batches"]):
            save_pdf2d_vals[k][b0:b1, :, :] = hist.astype(np.float32)

        if lnp_outname is not None:
            for bi in range(B):
                idx = kernel_result["retained_local_indices"][bi]

                if lnp_npts is not None and lnp_npts < len(idx):
                    idx = idx[:lnp_npts]

                e = b0 + bi
                if fit_model_block_size:
                    lnp_save = kernel_result["retained_lnp"][bi][: len(idx)]
                    chi2_save = kernel_result["retained_chi2"][bi][: len(idx)]
                else:
                    lnp_save = kernel_result["lnp"][bi, idx]
                    chi2_save = kernel_result["chi2"][bi, idx]
                save_lnp_vals.append([
                    e,
                    np.array(g0_indxs[idx], dtype=np.int64),
                    np.array(lnp_save, dtype=np.float32),
                    np.array(chi2_save, dtype=np.float32),
                    np.array([Y[bi]]).T,
                ])

    if pdf1d_outname is not None:
        save_pdf1d(pdf1d_outname, save_pdf1d_vals, qnames)

    if pdf2d_outname is not None and len(pdf2d_qname_pairs) > 0:
        save_pdf2d(pdf2d_outname, save_pdf2d_vals, pdf2d_qname_pairs)

    if stats_outname is not None:
        save_stats(
            stats_outname,
            prev_result,
            best_vals,
            exp_vals,
            per_vals,
            chi2_vals,
            chi2_indx,
            lnp_vals,
            lnp_indx,
            best_specgrid_indx,
            total_log_norm,
            qnames,
            p,
            sedgrid.filters,
            sedgrid.lamb,
        )

    if lnp_outname is not None:
        save_lnp(lnp_outname, save_lnp_vals)

    profile_summary("Q_all_memory_batched profile summary", reset=True)
    return None


def Q_all_memory(
    prev_result,
    obs,
    sedgrid,
    obsmodel,
    qnames_in,
    p=[16.0, 50.0, 84.0],
    gridbackend="cache",
    max_nbins=200,
    stats_outname=None,
    pdf1d_outname=None,
    pdf2d_outname=None,
    pdf2d_param_list=None,
    grid_info_dict=None,
    lnp_outname=None,
    lnp_npts=None,
    save_every_npts=None,
    threshold=-40,
    resume=False,
    use_full_cov_matrix=True,
    do_not_normalize=False,
    compute_percentiles=True,
):
    """
    Fit each star, calculate various fit statistics, and output them to files.
    All done in one function for speed and ability to resume partially completed runs.

    Parameters
    ----------
    prev_result : dict
        previous results to include in the output summary table
        usually basic data on each source
    obs : Observation object instance
        observation catalog
    sedgrid : str or grid.SEDgrid instance
        model grid
    obsmodel : beast noisemodel instance
        noise model data
    qnames : list
        names of quantities
    p : array-like
        list of percentile values
    gridbackend : str or grid.GridBackend
        backend to use to load the grid if necessary (memory, cache, hdf)
        (see beast.core.grid)
    max_nbins : int (default=200)
        maxiumum number of bins to use for the 1D likelihood calculations
    save_every_npts : int
        set to save the files below (if set) every n stars
        a requirement for recovering from partially complete runs
    resume : bool
        set to designate this run is resuming a partially complete run
    use_full_cov_matrix : bool
        set to use the full covariance matrix if it is present in the
        noise model file
    stats_outname : str
        set to output the stats file into a FITS file with extensions
    pdf1d_outname : str
        set to output the 1D PDFs into a FITS file with extensions
    pdf2d_outname : str
        set to output the 2D PDFs into a FITS file with extensions
    pdf2d_param_list : list of strs or None
        set to the parameters for which to make the 2D PDFs
    grid_info_dict : dict
        Set to override the mins/maxes of the 1dpdfs, and the number of
        unique values
    lnp_outname : str
        set to output the sparse likelihoods into a (usually HDF5) file
    threshold : float
        value above which to use/save for the lnps (defines the sparse likelihood)
    lnp_npts : int
        set to a number to output a random sampling of the lnp points above
        the threshold. Otherwise, the full sparse likelihood is output.
    do_not_normalize: bool
        Do not normalize the prior weights before applying them. This
        should have no effect on the final outcome when using only a
        single grid, but is essential when using the subgridding
        approach.

    Returns
    -------
    N/A
    """

    with profile_stage("loading physics/noise grids", detail="Q_all_memory sedgrid"):
        if isinstance(sedgrid, str):
            g0 = grid.SEDGrid(sedgrid, backend=gridbackend)
        else:
            g0 = sedgrid

    # remove weights that are less than zero
    (g0_indxs,) = np.where(g0["weight"] > 0.0)

    for i, cfilter in enumerate(sedgrid.filters):
        (incomp_indxs,) = np.where(obsmodel["completeness"][:, i] <= 0.0)
        if len(incomp_indxs) > 0:
            raise ValueError(
                "models with zero completeness present in the observation model"
            )

    g0_weights = np.log(g0["weight"][g0_indxs])
    if not do_not_normalize:
        # this variable used on the next line, so is used regardless of what flake8 says
        g0_weights_sum = np.log(g0["weight"][g0_indxs].sum())  # noqa: F841
        g0_weights = numexpr.evaluate("g0_weights - g0_weights_sum")

    if len(g0["weight"]) != len(g0_indxs):
        print("orig/g0_indxs", len(g0["weight"]), len(g0_indxs))
        warnings.warn("some zero weight models exist")

    # get the model SEDs
    if hasattr(g0.seds, "read"):
        _seds = g0.seds.read()
    else:
        _seds = g0.seds

    # links to errors and biases
    ast_error = obsmodel["error"]
    ast_bias = obsmodel["bias"]

    # if the ast file includes the full covariance matrices, make links
    full_cov_mat = False
    if (
        use_full_cov_matrix
        & ("q_norm" in obsmodel.keys())
        & ("icov_diag" in obsmodel.keys())
        & ("icov_offdiag" in obsmodel.keys())
    ):
        full_cov_mat = True
        ast_q_norm = obsmodel["q_norm"]
        ast_icov_diag = obsmodel["icov_diag"]
        two_ast_icov_offdiag = 2.0 * obsmodel["icov_offdiag"]
    else:
        ast_ivar = 1.0 / np.asfortranarray(ast_error) ** 2

    if full_cov_mat:
        print("using full covariance matrix")
    else:
        print("not using full covariance matrix")

    # number of observed SEDs to fit
    nobs = len(obs)

    # augment the qnames to include the *full* model SED
    #  by this it means the physical model flux plus the noise model bias term
    qnames = qnames_in
    filters = sedgrid.filters
    for i, cfilter in enumerate(filters):
        qnames.append("symlog" + cfilter + "_wd_bias")

    # create the full model fluxes for later use
    #   save as symmetric log, since the fluxes can be negative
    model_seds_with_bias = np.asfortranarray(_seds + ast_bias)
    # full_model_flux = np.sign(logtempseds) * np.log10(1 + np.abs(logtempseds * math.log(10)))
    full_model_flux = symlog(model_seds_with_bias)

    p = list(p) if compute_percentiles else []

    # setup the arrays to temp store the results
    n_qnames = len(qnames)
    n_pers = len(p)
    best_vals = np.zeros((nobs, n_qnames))
    exp_vals = np.zeros((nobs, n_qnames))
    per_vals = np.zeros((nobs, n_qnames, n_pers))
    chi2_vals = np.zeros(nobs)
    chi2_indx = np.zeros(nobs)
    lnp_vals = np.zeros(nobs)
    lnp_indx = np.zeros(nobs)
    best_specgrid_indx = np.zeros(nobs)
    total_log_norm = np.zeros(nobs)

    # variable to save the lnp files
    save_lnp_vals = []

    # setup the mapping for the 1D PDFs
    need_pdf1d = (pdf1d_outname is not None) or compute_percentiles
    fast_pdf1d_objs = []
    save_pdf1d_vals = []

    # make 1D PDF objects
    if need_pdf1d:
        with profile_stage("PDF1D/PDF2D generation", detail="setup 1D PDF bins"):
            for qname in qnames:

                # get bin properties
                qname_vals, nbins, logspacing, minval, maxval, uniqvals = setup_param_bins(
                    qname, max_nbins, g0, full_model_flux, filters, grid_info_dict
                )

                # generate the fast 1d pdf mapping
                _tpdf1d = pdf1d(
                    qname_vals,
                    nbins,
                    logspacing=logspacing,
                    minval=minval,
                    maxval=maxval,
                    uniqvals=uniqvals,
                )
                fast_pdf1d_objs.append(_tpdf1d)

                # setup the arrays to save the 1d PDFs
                if pdf1d_outname is not None:
                    save_pdf1d_vals.append(np.zeros((nobs + 1, nbins)))
                    save_pdf1d_vals[-1][-1, :] = _tpdf1d.bin_vals

    # if chosen, make 2D PDFs
    if pdf2d_outname is not None:
        with profile_stage("PDF1D/PDF2D generation", detail="setup 2D PDF bins"):
            # setup the 2D PDFs
            _pdf2d_params = [
                qname
                for qname in qnames
                if qname in pdf2d_param_list and len(np.unique(g0[qname])) > 1
            ]
            _n_params = len(_pdf2d_params)
            pdf2d_qname_pairs = [
                _pdf2d_params[i] + "+" + _pdf2d_params[j]
                for i in range(_n_params)
                for j in range(i + 1, _n_params)
            ]
            fast_pdf2d_objs = []
            save_pdf2d_vals = []

            # make 2D PDF objects
            for qname_pair in pdf2d_qname_pairs:
                qname_1, qname_2 = qname_pair.split("+")

                # get bin properties
                (
                    qname_vals_p1,
                    nbins_p1,
                    logspacing_p1,
                    minval_p1,
                    maxval_p1,
                    uniqvals_p1,
                ) = setup_param_bins(
                    qname_1, max_nbins, g0, full_model_flux, filters, grid_info_dict
                )
                (
                    qname_vals_p2,
                    nbins_p2,
                    logspacing_p2,
                    minval_p2,
                    maxval_p2,
                    uniqvals_p2,
                ) = setup_param_bins(
                    qname_2, max_nbins, g0, full_model_flux, filters, grid_info_dict
                )

                # make 2D PDF
                _tpdf2d = pdf2d(
                    qname_vals_p1,
                    qname_vals_p2,
                    nbins_p1,
                    nbins_p2,
                    logspacing_p1=logspacing_p1,
                    logspacing_p2=logspacing_p2,
                    minval_p1=minval_p1,
                    maxval_p1=maxval_p1,
                    minval_p2=minval_p2,
                    maxval_p2=maxval_p2,
                )
                fast_pdf2d_objs.append(_tpdf2d)
                # arrays for the PDFs and bins
                save_pdf2d_vals.append(np.zeros((nobs + 2, nbins_p1, nbins_p2)))
                save_pdf2d_vals[-1][-2, :, :] = np.tile(
                    _tpdf2d.bin_vals_p1, (nbins_p2, 1)
                ).T
                save_pdf2d_vals[-1][-1, :, :] = np.tile(
                    _tpdf2d.bin_vals_p2, (nbins_p1, 1)
                )

    # if this is a resume job, read in the already computed stats and
    #     fill the variables
    # also - find the start position for the resumed run
    if resume:
        stats_table = Table.read(stats_outname, hdu=1)

        for k, qname in enumerate(qnames):
            best_vals[:, k] = stats_table["{0:s}_Best".format(qname)]
            exp_vals[:, k] = stats_table["{0:s}_Exp".format(qname)]
            for i, pval in enumerate(p):
                per_vals[:, k, i] = stats_table["{0:s}_p{1:d}".format(qname, int(pval))]

        chi2_vals = stats_table["chi2min"]
        chi2_indx = stats_table["chi2min_indx"]
        lnp_vals = stats_table["Pmax"]
        lnp_indx = stats_table["Pmax_indx"]
        best_specgrid_indx = stats_table["specgrid_indx"]

        (indxs,) = np.where(stats_table["Pmax"] != 0.0)
        start_pos = max(indxs) + 1
        print(
            "resuming run with start indx = "
            + str(start_pos)
            + " out of "
            + str(len(stats_table["Pmax"]))
        )

        # read in the already computed 1D PDFs
        if pdf1d_outname is not None:
            print("restoring the already computed 1D PDFs from " + pdf1d_outname)
            with fits.open(pdf1d_outname) as hdulist:
                for k in range(len(qnames)):
                    save_pdf1d_vals[k] = hdulist[k + 1].data

        # read in the already computed 2D PDFs
        if pdf2d_outname is not None:
            print("restoring the already computed 2D PDFs from " + pdf2d_outname)
            with fits.open(pdf2d_outname) as hdulist:
                for k in range(len(pdf2d_qname_pairs)):
                    save_pdf2d_vals[k] = hdulist[k + 1].data

    else:
        start_pos = 0

        # setup a new lnp file
        if lnp_outname is not None:
            with profile_stage("output writing", detail=lnp_outname):
                outfile = tables.open_file(lnp_outname, "w")
                # Save wavelengths in root, remember #n_stars = root._v_nchildren -1
                outfile.create_array(outfile.root, "grid_waves", g0.lamb[:])
                filters = obs.getFilters()
                outfile.create_array(outfile.root, "obs_filters", filters[:])
                outfile.close()

    # loop over the objects and get all the requested quantities
    g0_specgrid_indx = g0["specgrid_indx"]
    _p = np.asarray(p, dtype=float)

    it = tqdm(
        islice(obs.enumobs(), int(start_pos), None),
        total=len(obs) - start_pos,
        desc="Calculating Lnp/Stats",
    )
    for e, obj in it:
        # calculate the full nD posterior
        sed = obj

        cur_mask = sed == 0
        # need an alternate way to generate the mask as zeros can be
        # valid values in the observed SED (KDG 29 Jan 2016)
        # currently, set mask to False always
        cur_mask[:] = False

        with profile_stage("likelihood computation"):
            if full_cov_mat:
                lnp, chi2 = N_covar_logLikelihood(
                    sed,
                    model_seds_with_bias,
                    ast_q_norm,
                    ast_icov_diag,
                    two_ast_icov_offdiag,
                    lnp_threshold=abs(threshold),
                )
            else:
                lnp, chi2 = N_logLikelihood_NM(
                    sed,
                    model_seds_with_bias,
                    ast_ivar,
                    mask=cur_mask,
                    lnp_threshold=abs(threshold),
                )

            lnp = lnp[g0_indxs]
            chi2 = chi2[g0_indxs]

        with profile_stage("posterior normalization/top-k"):
            # lnp = numexpr.evaluate('lnp + g0_weights')
            lnp += g0_weights  # multiply by the prior weights (sum in log space)

            (indx,) = np.where((lnp - max(lnp[np.isfinite(lnp)])) > threshold)

            # now generate the sparse likelihood (remove later if this works
            #       by updating code below)
            #   checked if changing to the full likelihood speeds things up
            #       - the answer is no
            #   and is likely related to the switch here to the sparse
            #       likelihood for the weight calculation
            lnps = lnp[indx]
            chi2s = chi2[indx]

            # log_norm = np.log(getNorm_lnP(lnps))
            # if not np.isfinite(log_norm):
            #    log_norm = lnps.max()
            log_norm = lnps.max()
            weights = np.exp(lnps - log_norm)

            # normalize the weights make sure they sum to one
            #   needed for np.random.choice
            weight_sum = np.sum(weights)
            weights /= weight_sum

        # save the current set of lnps
        if lnp_outname is not None:
            if lnp_npts is not None:
                if lnp_npts < len(indx):
                    rindx = np.random.choice(indx, size=lnp_npts, replace=False)
                if lnp_npts >= len(indx):
                    rindx = indx
            else:
                rindx = indx
            save_lnp_vals.append(
                [
                    e,
                    np.array(g0_indxs[rindx], dtype=np.int64),
                    np.array(lnp[rindx], dtype=np.float32),
                    np.array(chi2[rindx], dtype=np.float32),
                    np.array([sed]).T,
                ]
            )

        # To merge the stats for different subgrids, we need the total
        # weight of a grid, which is sum(exp(lnps)). Since sum(exp(lnps
        # - log_norm - log(weight_sum))) = 1, the relative weight of
        # each subgrid will be exp(log_norm + log(weight_sum)).
        # Therefore, we also store the following quantity:
        with profile_stage("summary statistics"):
            total_log_norm[e] = log_norm + np.log(weight_sum)

            # index to the full model grid for the best fit values
            best_full_indx = g0_indxs[indx[weights.argmax()]]

            # index to the spectral grid
            best_specgrid_indx[e] = g0_specgrid_indx[best_full_indx]

            # goodness of fit quantities
            chi2_vals[e] = chi2s.min()
            chi2_indx[e] = g0_indxs[indx[chi2s.argmin()]]
            lnp_vals[e] = lnps.max()
            lnp_indx[e] = best_full_indx

        # calculate quantities for individual parameters:
        # best value, expectation value, 1D PDF, percentiles
        for k, qname in enumerate(qnames):
            if "_bias" in qname:
                fname = (qname.replace("_wd_bias", "")).replace("symlog", "")
                q = full_model_flux[:, filters.index(fname)]
            else:
                q = g0[qname]

            with profile_stage("summary statistics"):
                # best value
                best_vals[e, k] = q[best_full_indx]

                # expectation value
                exp_vals[e, k] = expectation(q[g0_indxs[indx]], weights=weights)

            if need_pdf1d:
                with profile_stage("PDF1D/PDF2D generation"):
                    # percentile values and/or saved 1D PDFs
                    pdf1d_bins, pdf1d_vals = fast_pdf1d_objs[k].gen1d(
                        g0_indxs[indx], weights
                    )

                    if pdf1d_outname is not None:
                        save_pdf1d_vals[k][e, :] = pdf1d_vals
                    if compute_percentiles:
                        if pdf1d_vals.max() > 0:
                            # remove normalization to allow for post processing with
                            #   different distance runs (needed for the SMIDGE-SMC)
                            # pdf1d_vals /= pdf1d_vals.max()
                            per_vals[e, k, :] = percentile(
                                pdf1d_bins, _p, weights=pdf1d_vals
                            )
                        else:
                            per_vals[e, k, :] = np.zeros(len(p), dtype=float)

        # calculate 2D PDFs for the subset of parameter pairs
        if pdf2d_outname is not None:
            with profile_stage("PDF1D/PDF2D generation"):
                for k in range(len(pdf2d_qname_pairs)):
                    save_pdf2d_vals[k][e, :, :] = fast_pdf2d_objs[k].gen2d(
                        g0_indxs[indx], weights
                    )

        # incremental save (useful if job dies early to recover most
        #    of the computations)
        if save_every_npts is not None:
            if (e > 0) & (e % save_every_npts == 0):
                # save the 1D PDFs
                if pdf1d_outname is not None:
                    save_pdf1d(pdf1d_outname, save_pdf1d_vals, qnames)

                # save the 2D PDFs
                if pdf2d_outname is not None:
                    save_pdf2d(pdf2d_outname, save_pdf2d_vals, pdf2d_qname_pairs)

                # save the stats/catalog
                if stats_outname is not None:
                    save_stats(
                        stats_outname,
                        prev_result,
                        best_vals,
                        exp_vals,
                        per_vals,
                        chi2_vals,
                        chi2_indx,
                        lnp_vals,
                        lnp_indx,
                        best_specgrid_indx,
                        total_log_norm,
                        qnames,
                        p,
                        sedgrid.filters,
                        sedgrid.lamb,
                    )

                # save the lnps
                if lnp_outname is not None:
                    save_lnp(lnp_outname, save_lnp_vals)
                    save_lnp_vals = []

    # do the final save of everything (or the last set for the lnp values)

    # save the 1D PDFs
    if pdf1d_outname is not None:
        save_pdf1d(pdf1d_outname, save_pdf1d_vals, qnames)

    # save the 2D PDFs
    if pdf2d_outname is not None:
        save_pdf2d(pdf2d_outname, save_pdf2d_vals, pdf2d_qname_pairs)

    # save the stats/catalog
    if stats_outname is not None:
        save_stats(
            stats_outname,
            prev_result,
            best_vals,
            exp_vals,
            per_vals,
            chi2_vals,
            chi2_indx,
            lnp_vals,
            lnp_indx,
            best_specgrid_indx,
            total_log_norm,
            qnames,
            p,
            sedgrid.filters,
            sedgrid.lamb,
        )

    # save the lnps
    if lnp_outname is not None:
        save_lnp(lnp_outname, save_lnp_vals)

    profile_summary("Q_all_memory profile summary", reset=True)


def IAU_names_and_extra_info(obsdata, surveyname="PHAT", extraInfo=False):
    """
    Generates IAU approved names for the data using RA & DEC
    and extra information about the sources (ra, dec, photometry, etc.)

    Parameters
    ----------
    obsdata : class
        observations data
    surveyname : str
        name of survey [default = 'PHAT']
    extraInfo : bool
        set to get the HST specific PHAT software reduced survey information

    Returns
    -------
    r : dict
        A dict with a (name, ndarray) pair
    """
    r = {}

    go_name = False
    if "ra" in list(obsdata.data.keys()):
        go_name = True
        ra_str = "ra"
        dec_str = "dec"

    if "RA" in list(obsdata.data.keys()):
        go_name = True
        ra_str = "RA"
        dec_str = "DEC"

    if go_name:
        # generate the IAU names

        coords = ap_SkyCoord(
            obsdata.data[ra_str],
            obsdata.data[dec_str],
            unit=ap_units.degree,
            frame="icrs",
        )

        ra_string = coords.ra.to_string(
            unit=ap_units.hourangle,
            sep="",
            precision=4,
            alwayssign=False,
            pad=True,
        )

        dec_string = coords.dec.to_string(
            sep="",
            precision=3,
            alwayssign=True,
            pad=True,
        )

        r["Name"] = np.array([f"{surveyname} J{ra}{dec}" for ra, dec in zip(ra_string, dec_string)],dtype=str)
    

        # other useful information
        r["RA"] = obsdata.data[ra_str]
        r["DEC"] = obsdata.data[dec_str]
        if extraInfo:
            r["field"] = obsdata.data["field"]
            r["inside_brick"] = obsdata.data["inside_brick"]
            r["inside_chipgap"] = obsdata.data["inside_chipgap"]

    else:
        r["Name"] = ["noname" for x in range(len(obsdata))]

    # include the observed filter fluxes
    for k, filtername in enumerate(obsdata.filters):
        obsfiltname = obsdata.filter_aliases[filtername]
        r[filtername] = (obsdata.data[obsfiltname] * obsdata.vega_flux[k]).astype(float)

    # if running a simulation, propagate beast_idx numbers to resort to input obs file
    if "beast_idx" in list(obsdata.data.keys()):
        r["beast_idx"] = obsdata.data["beast_idx"]

    return r


def summary_table_memory(
    obs,
    noisemodel,
    sedgrid,
    keys=None,
    gridbackend="cache",
    threshold=-40.0,
    save_every_npts=None,
    lnp_npts=None,
    resume=False,
    max_nbins=200,
    stats_outname=None,
    pdf1d_outname=None,
    pdf2d_outname=None,
    pdf2d_param_list=None,
    grid_info_dict=None,
    lnp_outname=None,
    use_full_cov_matrix=True,
    surveyname=None,
    extraInfo=False,
    do_not_normalize=False,
    fit_use_batched=False,
    fit_use_topk=False,
    fit_star_batch_size=512,
    fit_topk_mass_target=0.999,
    fit_topk_ess_target=128.0,
    fit_topk_kmin=32,
    fit_topk_kmax=2048,
    fit_topk_kquantile=0.95,
    fit_model_block_size=0,
    backend="numpy",
    compute_percentiles=True,
):
    """
    Do the fitting in memory

    Parameters
    ----------
    obs : Observation object instance
        observation catalog
    noisemodel : beast noisemodel instance
        noise model data
    sedgrid : str or grid.SEDgrid instance
        model grid
    keys : str or list of str
        if str - name of the quantity or expression to evaluate from the grid table
        if list - list of quantities or expresions
    gridbackend : str or grid.GridBackend
        backend to use to load the grid if necessary (memory, cache, hdf)
        (see beast.core.grid)
    save_every_npts : integer
        set to save the files below (if set) every n stars
        a requirement for recovering from partially complete runs
    resume : bool
        set to designate this run is resuming a partially complete run
    use_full_cov_matrix : bool
        set to use the full covariance matrix if it is present in the
        noise model file
    max_nbins : int (default=200)
        maxiumum number of bins to use for the 1D likelihood calculations
    stats_outname : str
        set to output the stats file into a FITS file with extensions
    pdf1d_outname : str
        set to output the 1D PDFs into a FITS file with extensions
    pdf2d_outname : str
        set to output the 2D PDFs into a FITS file with extensions
    pdf2d_param_list : list of strings or None
        set to the parameters for which to make the 2D PDFs
    grid_info_dict : dict
        Set to override the mins/maxes of the 1dpdfs, and the number of
        unique values.
    lnp_outname : str
        set to output the sparse likelihoods into a (usually HDF5) file
    threshold : float
        value above which to use/save for the lnps (defines the sparse likelihood)
    lnp_npts : int
        set to a number to output a random sampling of the lnp points above
        the threshold.  otherwise, the full sparse likelihood is output
    surveyname : str
          name of survey [default = 'PHAT']
    extraInfo : bool
        set to get extra information, such as IAU name, brick, field, etc.
    do_not_normalize : bool
        Do not normalize the prior weights before applying them. This
        should have no effect on the final outcome when using only a
        single grid, but is essential when using the subgridding
        approach.
    backend : {"numpy", "jax"}
        numerical backend for the batched fitting kernel
    fit_model_block_size : int
        If positive and ``fit_use_batched`` is true, stream the diagonal
        likelihood over model blocks of this size instead of materializing the
        full star-batch by model-grid posterior matrix.

    Returns
    -------
    N/A

    """

    if isinstance(sedgrid, str):
        g0 = grid.SEDGrid(sedgrid, backend=gridbackend)
    else:
        g0 = sedgrid

    if keys is None:
        keys = list(g0.keys())

    # make sure keys are real keys
    skip_keys = "osl keep weight grid_weight prior_weight fullgrid_idx stage specgrid_indx phase eep".split()
    keys = [k for k in keys if k not in skip_keys]

    for key in keys:
        if not (key in list(g0.keys())):
            raise KeyError('Key "{0}" not recognized'.format(key))

    # make sure there are 2D PDF params if needed
    if (pdf2d_outname is not None) and (pdf2d_param_list is None):
        raise KeyError("pdf2d_param_list cannot be None if saving 2D PDFs")

    # generate an IAU complient name for each source and add other inform
    res = IAU_names_and_extra_info(obs, surveyname=surveyname, extraInfo=False)

    # --------------------------------------------
    # choose fitting core
    # --------------------------------------------
    if fit_use_batched:
        Q_all_memory_batched(
            res,
            obs,
            g0,
            noisemodel,
            keys,
            p=[16.0, 50.0, 84.0],
            compute_percentiles=compute_percentiles,
            resume=resume,
            threshold=threshold,
            save_every_npts=save_every_npts,
            lnp_npts=lnp_npts,
            max_nbins=max_nbins,
            stats_outname=stats_outname,
            pdf1d_outname=pdf1d_outname,
            pdf2d_outname=pdf2d_outname,
            pdf2d_param_list=pdf2d_param_list,
            grid_info_dict=grid_info_dict,
            lnp_outname=lnp_outname,
            use_full_cov_matrix=use_full_cov_matrix,
            do_not_normalize=do_not_normalize,
            fit_use_topk=fit_use_topk,
            fit_star_batch_size=fit_star_batch_size,
            fit_topk_mass_target=fit_topk_mass_target,
            fit_topk_ess_target=fit_topk_ess_target,
            fit_topk_kmin=fit_topk_kmin,
            fit_topk_kmax=fit_topk_kmax,
            fit_topk_kquantile=fit_topk_kquantile,
            fit_model_block_size=fit_model_block_size,
            backend=backend,
        )
    else:
        Q_all_memory(
            res,
            obs,
            g0,
            noisemodel,
            keys,
            p=[16.0, 50.0, 84.0],
            compute_percentiles=compute_percentiles,
            resume=resume,
            threshold=threshold,
            save_every_npts=save_every_npts,
            lnp_npts=lnp_npts,
            max_nbins=max_nbins,
            stats_outname=stats_outname,
            pdf1d_outname=pdf1d_outname,
            pdf2d_outname=pdf2d_outname,
            pdf2d_param_list=pdf2d_param_list,
            grid_info_dict=grid_info_dict,
            lnp_outname=lnp_outname,
            use_full_cov_matrix=use_full_cov_matrix,
            do_not_normalize=do_not_normalize,
        )

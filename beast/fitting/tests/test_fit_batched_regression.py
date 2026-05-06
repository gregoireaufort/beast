import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from beast.fitting.fit import (
    Q_all_memory,
    Q_all_memory_batched,
    q_all_memory_batched_kernel,
)
from beast.physicsmodel.grid import SEDGrid


class TinyObservations:
    def __init__(self, fluxes, filters):
        self._fluxes = np.asarray(fluxes, dtype=np.float64)
        self.filters = list(filters)

    def __len__(self):
        return self._fluxes.shape[0]

    def enumobs(self):
        for i, flux in enumerate(self._fluxes):
            yield i, flux

    def getFilters(self):
        return self.filters


def _tiny_fit_inputs():
    filters = ["F1", "F2", "F3"]

    seds = np.array(
        [
            [1.00, 2.00, 3.00],
            [1.30, 2.15, 2.85],
            [2.20, 1.75, 2.40],
            [3.20, 3.00, 1.20],
            [4.10, 3.60, 1.60],
        ],
        dtype=np.float64,
    )

    grid_table = Table(
        {
            "M_ini": np.array([1.0, 1.5, 2.0, 3.0, 4.0]),
            "Av": np.array([0.1, 0.2, 0.5, 0.9, 1.4]),
            "weight": np.array([1.0, 0.7, 0.5, 0.9, 0.4]),
            "specgrid_indx": np.arange(len(seds), dtype=np.int64),
        }
    )
    sedgrid = SEDGrid(
        np.array([1.0, 2.0, 3.0]),
        seds=seds,
        grid=grid_table,
        header={"filters": " ".join(filters)},
        backend="memory",
    )

    obs = TinyObservations(
        np.array(
            [
                [1.05, 2.02, 2.95],
                [3.15, 2.95, 1.25],
                [2.00, 1.85, 2.55],
            ],
            dtype=np.float64,
        ),
        filters,
    )

    noisemodel = {
        "bias": np.zeros_like(seds),
        "error": np.full_like(seds, 0.35),
        "completeness": np.ones_like(seds),
    }
    prev_result = {"Name": np.array([f"star_{i}" for i in range(len(obs))])}
    return prev_result, obs, sedgrid, noisemodel


def _run_original(tmp_path, p=(16.0, 50.0, 84.0)):
    tmp_path.mkdir(parents=True, exist_ok=True)
    prev_result, obs, sedgrid, noisemodel = _tiny_fit_inputs()
    stats_path = tmp_path / "original_stats.fits"
    pdf1d_path = tmp_path / "original_pdf1d.fits"

    Q_all_memory(
        prev_result,
        obs,
        sedgrid,
        noisemodel,
        ["M_ini", "Av"],
        p=list(p),
        threshold=-40.0,
        max_nbins=10,
        stats_outname=str(stats_path),
        pdf1d_outname=str(pdf1d_path),
        use_full_cov_matrix=False,
        do_not_normalize=True,
    )

    return Table.read(stats_path, hdu=1), fits.open(pdf1d_path)


def _run_batched(tmp_path, p=(16.0, 50.0, 84.0), **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    prev_result, obs, sedgrid, noisemodel = _tiny_fit_inputs()
    stats_path = tmp_path / "batched_stats.fits"
    pdf1d_path = tmp_path / "batched_pdf1d.fits"
    fit_use_topk = kwargs.pop("fit_use_topk", False)

    Q_all_memory_batched(
        prev_result,
        obs,
        sedgrid,
        noisemodel,
        ["M_ini", "Av"],
        p=tuple(p),
        threshold=-40.0,
        max_nbins=10,
        stats_outname=str(stats_path),
        pdf1d_outname=str(pdf1d_path),
        use_full_cov_matrix=False,
        do_not_normalize=True,
        fit_use_topk=fit_use_topk,
        fit_star_batch_size=2,
        **kwargs,
    )

    return Table.read(stats_path, hdu=1), fits.open(pdf1d_path)


def test_q_all_memory_batched_matches_original_stats(tmp_path):
    original, original_pdf = _run_original(tmp_path)
    batched, batched_pdf = _run_batched(tmp_path)

    float_cols = [
        "chi2min",
        "Pmax",
        "M_ini_Exp",
        "Av_Exp",
        "M_ini_p16",
        "M_ini_p50",
        "M_ini_p84",
        "Av_p16",
        "Av_p50",
        "Av_p84",
    ]
    for col in float_cols:
        np.testing.assert_allclose(
            batched[col],
            original[col],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"{col} differs between original and batched fitting",
        )

    for col in ["Pmax_indx", "chi2min_indx", "specgrid_indx"]:
        np.testing.assert_array_equal(
            batched[col],
            original[col],
            err_msg=f"{col} differs between original and batched fitting",
        )

    for extname in ["M_ini", "Av"]:
        np.testing.assert_allclose(
            batched_pdf[extname].data[:-2],
            original_pdf[extname].data[:-1],
            rtol=1e-7,
            atol=1e-7,
            err_msg=f"{extname} 1D PDF differs between original and batched fitting",
        )
        np.testing.assert_allclose(
            batched_pdf[extname].data[-1],
            original_pdf[extname].data[-1],
            rtol=0.0,
            atol=1e-7,
            err_msg=f"{extname} 1D PDF bins differ between original and batched fitting",
        )

    original_pdf.close()
    batched_pdf.close()


def test_q_all_memory_stats_only_skips_pdf_percentiles_and_lnp(tmp_path):
    full_stats, full_pdf = _run_original(tmp_path / "full")
    prev_result, obs, sedgrid, noisemodel = _tiny_fit_inputs()
    stats_path = tmp_path / "lean" / "stats_only.fits"
    pdf1d_path = tmp_path / "lean" / "stats_only_pdf1d.fits"
    lnp_path = tmp_path / "lean" / "stats_only_lnp.hd5"
    stats_path.parent.mkdir()

    Q_all_memory(
        prev_result,
        obs,
        sedgrid,
        noisemodel,
        ["M_ini", "Av"],
        p=[16.0, 50.0, 84.0],
        threshold=-40.0,
        max_nbins=10,
        stats_outname=str(stats_path),
        pdf1d_outname=None,
        lnp_outname=None,
        use_full_cov_matrix=False,
        do_not_normalize=True,
        compute_percentiles=False,
    )

    stats = Table.read(stats_path, hdu=1)
    for col in ["chi2min", "Pmax", "Pmax_indx", "M_ini_Best", "M_ini_Exp", "Av_Exp"]:
        assert col in stats.colnames

    for col in ["chi2min", "Pmax", "M_ini_Best", "M_ini_Exp", "Av_Best", "Av_Exp"]:
        np.testing.assert_allclose(stats[col], full_stats[col], rtol=1e-12, atol=1e-12)

    for col in ["Pmax_indx", "chi2min_indx", "specgrid_indx"]:
        np.testing.assert_array_equal(stats[col], full_stats[col])

    for col in ["M_ini_p16", "M_ini_p50", "M_ini_p84", "Av_p16", "Av_p50", "Av_p84"]:
        assert col not in stats.colnames

    assert not pdf1d_path.exists()
    assert not lnp_path.exists()
    full_pdf.close()


def test_q_all_memory_batched_kernel_direct_small_arrays():
    _, obs, sedgrid, noisemodel = _tiny_fit_inputs()
    Y_batch = np.vstack([obj for _, obj in obs.enumobs()])
    model_seds_with_bias = sedgrid.seds + noisemodel["bias"]
    log_prior_weights = np.log(np.asarray(sedgrid["weight"]))
    ast_ivar = 1.0 / noisemodel["error"] ** 2
    q_param_arrays = np.vstack([np.asarray(sedgrid["M_ini"]), np.asarray(sedgrid["Av"])])
    pdf1d_bin_values = [q_param_arrays[0], q_param_arrays[1]]
    pdf1d_bin_indices = [
        np.searchsorted(pdf1d_bin_values[0], q_param_arrays[0], side="left"),
        np.searchsorted(pdf1d_bin_values[1], q_param_arrays[1], side="left"),
    ]

    result = q_all_memory_batched_kernel(
        Y_batch,
        model_seds_with_bias,
        log_prior_weights,
        q_param_arrays,
        pdf1d_bin_indices,
        pdf1d_bin_values,
        ast_ivar=ast_ivar,
        use_full_cov_matrix=False,
        threshold=-40.0,
        p=(16.0, 50.0, 84.0),
        fit_use_topk=False,
    )

    expected_chi2 = np.sum(
        (Y_batch[:, None, :] - model_seds_with_bias[None, :, :]) ** 2
        * ast_ivar[None, :, :],
        axis=2,
    )
    np.testing.assert_allclose(
        result["chi2_values"],
        expected_chi2.min(axis=1),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_array_equal(result["chi2_local_indices"], expected_chi2.argmin(axis=1))
    np.testing.assert_array_equal(result["lnp_local_indices"], np.array([0, 3, 2]))
    np.testing.assert_allclose(
        result["posterior_weights"].sum(axis=1), np.ones(Y_batch.shape[0])
    )
    np.testing.assert_allclose(
        result["best_vals"],
        np.array([[1.0, 0.1], [3.0, 0.9], [2.0, 0.5]]),
    )
    assert len(result["retained_local_indices"]) == Y_batch.shape[0]
    assert all(idx.ndim == 1 for idx in result["retained_local_indices"])


def test_q_all_memory_batched_kernel_jax_matches_numpy():
    pytest.importorskip("jax")

    _, obs, sedgrid, noisemodel = _tiny_fit_inputs()
    Y_batch = np.vstack([obj for _, obj in obs.enumobs()])
    model_seds_with_bias = sedgrid.seds + noisemodel["bias"]
    log_prior_weights = np.log(np.asarray(sedgrid["weight"]))
    ast_ivar = 1.0 / noisemodel["error"] ** 2
    q_param_arrays = np.vstack([np.asarray(sedgrid["M_ini"]), np.asarray(sedgrid["Av"])])
    pdf1d_bin_values = [q_param_arrays[0], q_param_arrays[1]]
    pdf1d_bin_indices = [
        np.searchsorted(pdf1d_bin_values[0], q_param_arrays[0], side="left"),
        np.searchsorted(pdf1d_bin_values[1], q_param_arrays[1], side="left"),
    ]

    common_kwargs = dict(
        Y_batch=Y_batch,
        model_seds_with_bias=model_seds_with_bias,
        log_prior_weights=log_prior_weights,
        q_param_arrays=q_param_arrays,
        pdf1d_bin_indices=pdf1d_bin_indices,
        pdf1d_bin_values=pdf1d_bin_values,
        ast_ivar=ast_ivar,
        use_full_cov_matrix=False,
        threshold=-40.0,
        p=(16.0, 50.0, 84.0),
        fit_use_topk=False,
    )

    numpy_result = q_all_memory_batched_kernel(**common_kwargs, backend="numpy")
    jax_result = q_all_memory_batched_kernel(**common_kwargs, backend="jax")

    for key in [
        "lnp",
        "chi2",
        "posterior_weights",
        "chi2_values",
        "lnp_values",
        "total_log_norm",
        "best_vals",
        "exp_vals",
        "per_vals",
    ]:
        np.testing.assert_allclose(
            jax_result[key],
            numpy_result[key],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"{key} differs between NumPy and JAX backends",
        )

    for key in ["chi2_local_indices", "lnp_local_indices"]:
        np.testing.assert_array_equal(jax_result[key], numpy_result[key])

    for jax_pdf, numpy_pdf in zip(
        jax_result["pdf1d_batches"], numpy_result["pdf1d_batches"]
    ):
        np.testing.assert_allclose(jax_pdf, numpy_pdf, rtol=1e-12, atol=1e-12)


def test_q_all_memory_batched_topk_mass_invariant(tmp_path):
    dense, dense_pdf = _run_batched(tmp_path / "dense")
    topk, topk_pdf = _run_batched(
        tmp_path / "topk",
        fit_use_topk=True,
        fit_topk_mass_target=1.0,
        fit_topk_ess_target=0.0,
        fit_topk_kmin=1,
        fit_topk_kmax=5,
        fit_topk_kquantile=1.0,
    )

    for col in ["chi2min", "Pmax", "Pmax_indx", "chi2min_indx"]:
        np.testing.assert_allclose(topk[col], dense[col], rtol=1e-12, atol=1e-12)

    dense_pdf.close()
    topk_pdf.close()

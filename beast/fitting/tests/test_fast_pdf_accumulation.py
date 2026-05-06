import numpy as np

from beast.fitting.pdf1d import pdf1d
from beast.fitting.pdf2d import pdf2d


def _reference_gen1d(pdf_obj, gindxs, weights):
    if pdf_obj.bad:
        return pdf_obj.bin_vals, np.zeros(pdf_obj.nbins)

    dense_weights = np.zeros(pdf_obj.n_gridvals)
    dense_weights[gindxs] = weights
    vals = np.zeros(pdf_obj.nbins)
    for i in range(pdf_obj.nbins):
        if len(pdf_obj.pdf_bin_indxs[i]) > 0:
            vals[i] = np.sum(dense_weights[pdf_obj.pdf_bin_indxs[i]])

    return pdf_obj.bin_vals, vals


def _reference_gen2d(pdf_obj, gindxs, weights):
    dense_weights = np.zeros(pdf_obj.n_gridvals)
    dense_weights[gindxs] = weights
    vals = np.zeros((pdf_obj.nbins_p1, pdf_obj.nbins_p2))
    for i in range(pdf_obj.nbins_p1):
        for j in range(pdf_obj.nbins_p2):
            if len(pdf_obj.pdf_bin_indxs[i][j]) > 0:
                vals[i, j] = np.sum(dense_weights[pdf_obj.pdf_bin_indxs[i][j]])

    return vals


def test_pdf1d_fast_accumulation_matches_reference():
    gridvals = np.array([0.1, 0.2, 0.2, 0.4, 0.7, 1.0, 1.0, 1.4, 1.8])
    gindxs = np.array([0, 2, 4, 5, 8])
    weights = np.array([0.05, 0.2, 0.3, 0.15, 0.3])

    pdf_obj = pdf1d(gridvals, 5)
    ref_bins, ref_vals = _reference_gen1d(pdf_obj, gindxs, weights)
    fast_bins, fast_vals = pdf_obj.gen1d(gindxs, weights)

    np.testing.assert_allclose(fast_bins, ref_bins, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(fast_vals, ref_vals, rtol=0.0, atol=1e-15)


def test_pdf1d_fast_accumulation_matches_reference_for_unique_bins():
    gridvals = np.array([1.0, 2.0, 2.0, 4.0, 8.0])
    gindxs = np.array([0, 1, 3, 4])
    weights = np.array([0.1, 0.2, 0.3, 0.4])

    pdf_obj = pdf1d(gridvals, 4, logspacing=True)
    ref_bins, ref_vals = _reference_gen1d(pdf_obj, gindxs, weights)
    fast_bins, fast_vals = pdf_obj.gen1d(gindxs, weights)

    np.testing.assert_allclose(fast_bins, ref_bins, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(fast_vals, ref_vals, rtol=0.0, atol=1e-15)


def test_pdf2d_fast_accumulation_matches_reference():
    gridvals_p1 = np.array([0.1, 0.2, 0.2, 0.4, 0.7, 1.0, 1.0, 1.4, 1.8])
    gridvals_p2 = np.array([5.0, 4.0, 3.5, 3.0, 2.2, 2.0, 1.5, 1.0, 0.5])
    gindxs = np.array([0, 2, 4, 5, 8])
    weights = np.array([0.05, 0.2, 0.3, 0.15, 0.3])

    pdf_obj = pdf2d(gridvals_p1, gridvals_p2, 5, 4)
    ref_vals = _reference_gen2d(pdf_obj, gindxs, weights)
    fast_vals = pdf_obj.gen2d(gindxs, weights)

    np.testing.assert_allclose(fast_vals, ref_vals, rtol=0.0, atol=1e-15)

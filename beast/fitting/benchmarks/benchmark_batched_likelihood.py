"""Benchmark dense vs blocked batched diagonal fitting kernels.

Run from the repository root, for example:

    python beast/fitting/benchmarks/benchmark_batched_likelihood.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _make_inputs(batch_size, n_models, n_filters, n_qparams, n_bins, seed):
    rng = np.random.default_rng(seed)

    model_seds_with_bias = rng.normal(1.0, 0.25, size=(n_models, n_filters))
    Y_batch = rng.normal(1.0, 0.25, size=(batch_size, n_filters))
    ast_ivar = rng.uniform(4.0, 25.0, size=(n_models, n_filters))
    log_prior_weights = np.log(rng.uniform(0.2, 1.0, size=n_models))
    q_param_arrays = rng.uniform(0.0, 1.0, size=(n_qparams, n_models))

    pdf1d_bin_values = [
        np.linspace(q_param_arrays[k].min(), q_param_arrays[k].max(), n_bins)
        for k in range(n_qparams)
    ]
    pdf1d_bin_indices = [
        np.clip(
            np.searchsorted(pdf1d_bin_values[k], q_param_arrays[k], side="left"),
            0,
            n_bins - 1,
        )
        for k in range(n_qparams)
    ]

    return {
        "Y_batch": Y_batch,
        "model_seds_with_bias": model_seds_with_bias,
        "log_prior_weights": log_prior_weights,
        "q_param_arrays": q_param_arrays,
        "pdf1d_bin_indices": pdf1d_bin_indices,
        "pdf1d_bin_values": pdf1d_bin_values,
        "ast_ivar": ast_ivar,
        "use_full_cov_matrix": False,
        "threshold": -40.0,
        "p": (16.0, 50.0, 84.0),
        "fit_use_topk": False,
    }


def _time_call(fn, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,8,32,128,512")
    parser.add_argument("--n-models", type=int, default=20000)
    parser.add_argument("--n-filters", type=int, default=6)
    parser.add_argument("--n-qparams", type=int, default=4)
    parser.add_argument("--n-bins", type=int, default=64)
    parser.add_argument("--model-block-size", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-jax", action="store_true")
    args = parser.parse_args()

    have_jax = False
    if not args.no_jax:
        try:
            import jax
        except ImportError:
            have_jax = False
        else:
            jax.config.update("jax_platform_name", "cpu")
            jax.config.update("jax_enable_x64", True)
            have_jax = True

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

    from beast.fitting.fit import (
        q_all_memory_batched_blocked_kernel,
        q_all_memory_batched_kernel,
    )

    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    print(
        "batch_size,n_models,n_filters,dense_numpy_seconds,"
        "blocked_numpy_seconds,blocked_jax_seconds,"
        "blocked_numpy_speedup_vs_dense,blocked_jax_speedup_vs_dense"
    )

    for batch_size in batch_sizes:
        inputs = _make_inputs(
            batch_size,
            args.n_models,
            args.n_filters,
            args.n_qparams,
            args.n_bins,
            args.seed + batch_size,
        )
        blocked_inputs = {
            key: value
            for key, value in inputs.items()
            if key not in {"use_full_cov_matrix", "fit_use_topk"}
        }

        def run_numpy():
            return q_all_memory_batched_kernel(**inputs, backend="numpy")

        def run_jax():
            return q_all_memory_batched_blocked_kernel(
                **blocked_inputs,
                backend="jax",
                model_block_size=args.model_block_size,
                retain_sparse=False,
            )

        def run_blocked_numpy():
            return q_all_memory_batched_blocked_kernel(
                **blocked_inputs,
                backend="numpy",
                model_block_size=args.model_block_size,
                retain_sparse=False,
            )

        run_numpy()
        run_blocked_numpy()
        if have_jax:
            run_jax()
        numpy_time = _time_call(run_numpy, args.repeats)
        blocked_numpy_time = _time_call(run_blocked_numpy, args.repeats)
        jax_time = _time_call(run_jax, args.repeats) if have_jax else np.nan
        blocked_numpy_speedup = (
            numpy_time / blocked_numpy_time if blocked_numpy_time > 0 else np.inf
        )
        jax_speedup = numpy_time / jax_time if have_jax and jax_time > 0 else np.nan

        print(
            f"{batch_size},{args.n_models},{args.n_filters},"
            f"{numpy_time:.6f},{blocked_numpy_time:.6f},{jax_time:.6f},"
            f"{blocked_numpy_speedup:.3f},{jax_speedup:.3f}"
        )


if __name__ == "__main__":
    main()

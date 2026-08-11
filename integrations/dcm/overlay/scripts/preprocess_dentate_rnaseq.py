"""Preprocess the Dentate Gyrus h5ad into discrete tokens for DCM training."""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/dentate/DentateGyrus.h5ad")
    parser.add_argument("--output", default="datasets/dentate/dentate_5000_bins32.h5ad")
    parser.add_argument("--num-genes", type=int, default=5000)
    parser.add_argument("--num-bins", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_bins < 2:
        raise ValueError("--num-bins must be at least 2")

    adata = sc.read_h5ad(args.input)
    x = adata.X
    if sparse.issparse(x):
        x = x.tocsr()
        x_log = x.copy()
        x_log.data = np.log1p(x_log.data)
        means = np.asarray(x_log.mean(axis=0)).ravel()
        sq_means = np.asarray(x_log.power(2).mean(axis=0)).ravel()
        variances = sq_means - means**2
    else:
        x_log = np.log1p(np.asarray(x))
        variances = x_log.var(axis=0)

    n_genes = min(args.num_genes, adata.n_vars)
    top_idx = np.argsort(variances)[-n_genes:]
    top_idx.sort()

    subset = adata[:, top_idx].copy()
    x_sub = subset.X
    if sparse.issparse(x_sub):
        dense = x_sub.toarray()
    else:
        dense = np.asarray(x_sub)
    dense = np.log1p(dense).astype(np.float32)

    positive = dense[dense > 0]
    if positive.size == 0:
        binned = np.zeros_like(dense, dtype=np.int16)
    else:
        quantiles = np.quantile(
            positive,
            np.linspace(0.0, 1.0, args.num_bins)[1:-1],
        )
        quantiles = np.unique(quantiles)
        binned = np.zeros_like(dense, dtype=np.int16)
        nonzero = dense > 0
        binned[nonzero] = np.searchsorted(
            quantiles,
            dense[nonzero],
            side="right",
        ).astype(np.int16) + 1
        binned = np.clip(binned, 0, args.num_bins - 1).astype(np.int16)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = ad.AnnData(
        X=binned,
        obs=subset.obs.copy(),
        var=subset.var.copy(),
        uns={
            "preprocessing": {
                "source": str(args.input),
                "num_genes": int(n_genes),
                "num_bins": int(args.num_bins),
                "method": "top_variance_log1p_quantile_bins",
                "zero_bin": 0,
                "valid_token_range": [0, int(args.num_bins - 1)],
            }
        },
    )
    result.write_h5ad(out)
    print(
        f"Wrote {out} shape={result.shape} "
        f"min={int(binned.min())} max={int(binned.max())} "
        f"sparsity={(binned == 0).mean():.2%}"
    )


if __name__ == "__main__":
    main()

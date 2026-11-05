#!/usr/bin/env python
"""Plot t-SNE of embeddings saved by evaluate_baseline.py.

Example
-------
python plot_tsne.py --embeddings_dir baseline_results/embeddings --out_dir tsne_plots --splits test
Accepts either the `embeddings` directory (every sub-directory holding <split>.npz files is plotted)
or a single such sub-directory.
"""
import argparse
import os
import sys

import numpy as np


def find_sets(root):
    """Return {name: dir} for each directory containing train/val/test .npz files."""
    sets = {}
    for d, _, files in os.walk(root):
        if any(f in ("train.npz", "val.npz", "test.npz") for f in files):
            sets[os.path.relpath(d, root) if d != root else os.path.basename(os.path.abspath(root))] = d
    return dict(sorted(sets.items()))


def load_split(d, splits, max_samples, rng):
    X, y, S, classes = [], [], [], None
    for s in splits:
        f = os.path.join(d, f"{s}.npz")
        if not os.path.isfile(f):
            continue
        z = np.load(f, allow_pickle=True)
        X.append(z["embeddings"]); y.append(z["labels"]); S += [s] * len(z["labels"])
        classes = [str(c) for c in z["classes"]]
    if not X:
        return None
    X, y, S = np.concatenate(X), np.concatenate(y), np.array(S)
    if max_samples and len(X) > max_samples:
        ix = rng.choice(len(X), max_samples, replace=False)
        X, y, S = X[ix], y[ix], S[ix]
    return X, y, S, classes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings_dir", required=True)
    ap.add_argument("--out_dir", required=True, help="where the plots (png + 2-D coordinates csv) are saved")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    ap.add_argument("--models", nargs="+", default=None, help="only embedding sets whose name contains one of these")
    ap.add_argument("--perplexity", type=float, default=30.0, help="auto-clamped to (n_samples-1)/3")
    ap.add_argument("--pca_dim", type=int, default=50, help="PCA before t-SNE (0 disables)")
    ap.add_argument("--max_samples", type=int, default=5000)
    ap.add_argument("--n_iter", type=int, default=1000)
    ap.add_argument("--marker_by_split", action="store_true", help="use different markers for train/val/test")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    sets = find_sets(a.embeddings_dir)
    if a.models:
        sets = {k: v for k, v in sets.items() if any(m in k for m in a.models)}
    if not sets:
        sys.exit(f"no embedding sets found in {a.embeddings_dir}")
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.RandomState(a.seed)
    markers = {"train": "o", "val": "s", "test": "^"}

    for name, d in sets.items():
        data = load_split(d, a.splits, a.max_samples, rng)
        if data is None:
            print(f"[skip] {name}: none of the requested splits present")
            continue
        X, y, S, classes = data
        if len(X) < 5:
            print(f"[skip] {name}: only {len(X)} samples")
            continue
        Xr = X.astype(np.float64)
        if a.pca_dim and Xr.shape[1] > a.pca_dim and len(Xr) > a.pca_dim:
            Xr = PCA(n_components=a.pca_dim, random_state=a.seed).fit_transform(Xr)
        perp = max(2.0, min(a.perplexity, (len(Xr) - 1) / 3))
        try:
            ts = TSNE(n_components=2, perplexity=perp, max_iter=a.n_iter, init="pca",
                      random_state=a.seed, learning_rate="auto")
        except TypeError:   # older scikit-learn
            ts = TSNE(n_components=2, perplexity=perp, n_iter=a.n_iter, init="pca", random_state=a.seed)
        Z = ts.fit_transform(Xr)

        fig, ax = plt.subplots(figsize=(7, 6))
        cmap = plt.get_cmap("tab10" if len(classes) <= 10 else "tab20")
        for ci, cname in enumerate(classes):
            for s in (np.unique(S) if a.marker_by_split else [None]):
                m = (y == ci) if s is None else ((y == ci) & (S == s))
                if m.any():
                    ax.scatter(Z[m, 0], Z[m, 1], s=18, alpha=0.8, color=cmap(ci % cmap.N),
                               marker=markers[s] if s else "o",
                               label=cname if (s is None or s == np.unique(S)[0]) else None)
        ax.set_title(f"t-SNE: {name}\n(splits: {','.join(a.splits)}, n={len(Z)}, perplexity={perp:.1f})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(title="class", fontsize=8, loc="best", markerscale=1.2)
        fig.tight_layout()
        base = os.path.join(a.out_dir, f"tsne__{name.replace(os.sep, '_')}")
        fig.savefig(f"{base}.{a.format}", dpi=a.dpi)
        plt.close(fig)
        np.savez_compressed(f"{base}_coords.npz", tsne=Z, labels=y, split=S, classes=np.array(classes))
        print(f"[ok] {name} -> {base}.{a.format}")


if __name__ == "__main__":
    main()

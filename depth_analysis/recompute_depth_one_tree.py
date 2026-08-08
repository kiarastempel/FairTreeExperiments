"""
Recompute actual tree depth and leaf count for the ONE-TREE scalarization/constrained
runs (chebyshev, weighted_combi, threshold_gain_s, backtracking).

Each best_results_seed_<seed>_min_samples_<pct>_max_depth_<d>.csv stores one row per
gamma value (column "gammas"). max_depth and the min_samples PERCENTAGE come from the
filename; the tree is reproduced with the same holdout split (random_state = seed) and
the same percentage->absolute conversion as one_tree_opt.py. Depth/leaves are read from
the fitted Node tree and appended as columns "actual_depth" and "num_leaves".

A safety check recomputes AUROC and compares it to the stored aurocs_test (column
"auroc_matches"). Depth convention: single-leaf tree = depth 0 (matches genome.depth).

Usage:
  python recompute_depth_one_tree.py --path_input <folder-with-csvs> --data Adult \
      --tree_variant chebyshev
"""
import os
import re
import sys
import glob
import copy
import argparse

import numpy as np
import pandas as pd

sys.path.append('../algos_two_trees')
sys.path.append('../../FairTree')

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from fair_classification_tree import FairClassificationTree
from get_data import data_loader_router


# depth / leaves on the Node tree (self.tree). Single leaf -> depth 0.
def tree_depth(node):
    if node is None or node.is_leaf:
        return 0
    return 1 + max(tree_depth(c) for c in node.children)


def tree_leaves(node):
    if node is None:
        return 0
    if node.is_leaf:
        return 1
    return sum(tree_leaves(c) for c in node.children)


FNAME = re.compile(r"best_results_seed_(\d+)_min_samples_(\d+)_max_depth_(\d+)")


def parse_name(fname):
    m = FNAME.search(os.path.basename(fname))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))  # seed, min_samples_pct, max_depth


def build_one_tree(X_train, y_train, s_train, tree_variant, max_depth, min_samples,
                   gamma, unprivileged_group, pos_outcome, max_h_y, min_h_s,
                   leaf_outcome_method="majority", predict_type="predict_proba"):
    """Rebuild one tree exactly as create_tree/one_tree_opt does."""
    X_fair = copy.deepcopy(X_train)
    X_fair['y'] = y_train
    cols = X_train.columns.tolist()
    sens = np.array(s_train.values).flatten().tolist()
    lom = "probability" if predict_type == "predict_proba" else leaf_outcome_method

    common = dict(data=X_fair, attributes=cols, idx_target=-1,
                  unprivileged_group=unprivileged_group, pos_outcome=pos_outcome,
                  threshold_binning=10, sensitive=sens, leaf_outcome=lom)

    if tree_variant == "threshold_gain_s":
        t = FairClassificationTree(**common, split_criterion="threshold_constraint",
                                   sens_threshold=gamma)
        t.fit(max_depth=max_depth, min_samples_leave=min_samples,
              tree_type="both", max_h_y=max_h_y, min_h_s=min_h_s)
    elif tree_variant == "weighted_combi":
        t = FairClassificationTree(**common, split_criterion="weighted_combi", gamma=gamma)
        t.fit(max_depth=max_depth, min_samples_leave=min_samples,
              tree_type="both", max_h_y=max_h_y, min_h_s=min_h_s)
    elif tree_variant == "chebyshev":
        t = FairClassificationTree(**common, split_criterion="chebyshev", gamma=gamma)
        t.fit(max_depth=max_depth, min_samples_leave=min_samples,
              tree_type="both", max_h_y=max_h_y, min_h_s=min_h_s)
    elif tree_variant == "backtracking":
        t = FairClassificationTree(**common, split_criterion="information_gain",
                                   sens_threshold=gamma, backtracking=True)
        t.fit(max_depth=max_depth, min_samples_leave=min_samples,
              tree_type="both", max_h_y=max_h_y, min_h_s=min_h_s)
    else:
        raise ValueError("unknown tree_variant: %s" % tree_variant)
    return t, cols


def process_file(csv_path, data, tree_variant, predict_type, leaf_outcome_method,
                 max_h_y, min_h_s, intersectional, check):
    parsed = parse_name(csv_path)
    if parsed is None:
        print("  skip (name not parseable):", os.path.basename(csv_path)); return
    seed, min_samples_pct, max_depth = parsed

    df = pd.read_csv(csv_path)

    X, y, s, unpriv, pos = data_loader_router(data, intersectional)
    X = X.applymap(lambda v: int(v) if isinstance(v, bool) else v).astype(np.float32)
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, pd.DataFrame(s), test_size=0.33, random_state=seed)

    min_samples = int(len(X_train) * min_samples_pct / 100)  # same conversion as opt script

    depths, leaves, ok = [], [], []
    for i, row in df.iterrows():
        print(i)
        gamma = row["gammas"]
        tree, cols = build_one_tree(X_train, y_train, s_train, tree_variant, max_depth,
                                    min_samples, gamma, unpriv, pos, max_h_y, min_h_s,
                                    leaf_outcome_method, predict_type)
        if tree.tree is None:
            depths.append(np.nan); leaves.append(np.nan); ok.append(np.nan); continue
        depths.append(tree_depth(tree.tree))
        leaves.append(tree_leaves(tree.tree))
        if check:
            Xte = copy.deepcopy(X_test); Xte['y'] = y_test
            pr = np.array(tree.predict(Xte[cols]))
            if predict_type == "predict_proba":
                pr = np.argmax(pr >= 0.5, axis=1)
            ok.append(bool(abs(roc_auc_score(y_test, pr) - row["aurocs_test"]) < 1e-6))

    df["actual_depth"] = depths
    df["num_leaves"] = leaves
    if check:
        df["auroc_matches"] = ok
    df.to_csv(csv_path, index=False)
    frac = np.mean([v for v in ok if not (isinstance(v, float) and np.isnan(v))]) if (check and ok) else float("nan")
    print(f"  seed {seed}: {len(df)} rows, mean depth {np.nanmean(depths):.2f}, "
          f"leaves {np.nanmean(leaves):.2f}" + (f", auroc match {frac:.2f}" if check else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path_input", required=True)
    ap.add_argument("--data", default="Adult")
    ap.add_argument("--tree_variant", default="chebyshev",
                    choices=["chebyshev", "weighted_combi", "threshold_gain_s", "backtracking"])
    ap.add_argument("--predict_type", default="predict_proba")
    ap.add_argument("--leaf_outcome_method", default="majority")
    ap.add_argument("--max_h_y", type=float, default=1.1)
    ap.add_argument("--min_h_s", type=float, default=0.0)
    ap.add_argument("--intersectional", action="store_true")
    ap.add_argument("--no_check", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.path_input, "best_results_seed_*.csv")))
    if not files:
        print("No best_results_seed_*.csv in", args.path_input); return
    print(f"{args.data} / {args.tree_variant}: {len(files)} file(s)")
    for f in files:
        process_file(f, args.data, args.tree_variant, args.predict_type,
                     args.leaf_outcome_method, args.max_h_y, args.min_h_s,
                     args.intersectional, check=not args.no_check)


if __name__ == "__main__":
    main()

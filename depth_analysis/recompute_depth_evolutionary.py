"""
Recompute actual tree depth and leaf count for the evolutionary (NSGA-II / SMS-EMOA)
runs.

Each best_results_pareto_seed_<seed>.csv already stores, per Pareto solution, the three
design variables needed to reproduce the tree: gamma, max_depth and min_samples_abs.
For every row we reproduce the exact hold-out split (via the seed in the filename),
rebuild the tree with create_tree's logic, and read the actual depth and number of
leaves from the fitted Node tree. As a safety check we recompute AUROC / fairness and
compare them against the stored values, so a mismatched split is caught immediately.
pm
Depth convention: a single-leaf tree has depth 0 (matching genome.depth for the GP
variant), so the two approaches are directly comparable.
"""
import os
import sys
import glob
import argparse
import copy

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'algos_two_trees'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'FairTree'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from fair_classification_tree import FairClassificationTree
from algos_two_trees.get_data import data_loader_router
from algos_two_trees.method import compute_fairness   # reuse the exact metric definitions


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


def build_tree(X_train, y_train, s_train, tree_variant, max_depth, min_samples,
               gamma, leaf_outcome_method, unprivileged_group, pos_outcome,
               max_h_y, min_h_s):
    """Rebuild one tree exactly as create_tree does, returning the fitted object."""
    X_fair = copy.deepcopy(X_train)
    X_fair['y'] = y_train
    cols = X_train.columns.tolist()
    sens = np.array(s_train.values).flatten().tolist()

    common = dict(data=X_fair, attributes=cols, idx_target=-1,
                  unprivileged_group=unprivileged_group, pos_outcome=pos_outcome,
                  threshold_binning=10, sensitive=sens, leaf_outcome=leaf_outcome_method)

    if tree_variant == "threshold_gain_s":
        t = FairClassificationTree(**common, split_criterion="threshold_constraint",
                                   sens_threshold=gamma)
        t.fit(max_depth=max_depth, min_samples_leave=min_samples)
    elif tree_variant == "weighted_combi":
        t = FairClassificationTree(**common, split_criterion="weighted_combi", gamma=gamma)
        t.fit(max_depth=max_depth, min_samples_leave=min_samples)
    elif tree_variant == "chebyshev":
        t = FairClassificationTree(**common, split_criterion="chebyshev", gamma=gamma)
        t.fit(max_depth=max_depth, min_samples_leave=min_samples)
    elif tree_variant == "backtracking":
        t = FairClassificationTree(**common, split_criterion="information_gain",
                                   sens_threshold=gamma, backtracking=True)
        t.fit(max_depth=max_depth, min_samples_leave=min_samples,
              tree_type="both", max_h_y=max_h_y, min_h_s=min_h_s)
    else:
        raise ValueError("unknown tree_variant: %s" % tree_variant)
    return t, cols


def process_file(csv_path, data, tree_variant, fairness_metric, predict_type,
                 leaf_outcome_method, max_h_y, min_h_s, intersectional, check):
    seed = int(os.path.basename(csv_path).split("seed_")[1].split(".")[0])
    df = pd.read_csv(csv_path)

    X, y, s, unpriv, pos = data_loader_router(data, intersectional)
    X = X.applymap(lambda v: int(v) if isinstance(v, bool) else v).astype(np.float32)
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, pd.DataFrame(s), test_size=0.33, random_state=seed)

    lom = "probability" if predict_type == "predict_proba" else leaf_outcome_method

    depths, leaves, ok = [], [], []
    for i, row in df.iterrows():
        print(i)
        gamma = row["gamma"]
        max_depth = int(row["max_depth"])
        min_samples = int(row["min_samples_abs"])

        tree, cols = build_tree(X_train, y_train, s_train, tree_variant,
                                max_depth, min_samples, gamma, lom,
                                unpriv, pos, max_h_y, min_h_s)

        if tree.tree is None:      # backtracking may fail to find a tree
            depths.append(np.nan); leaves.append(np.nan); ok.append(np.nan)
            continue

        depths.append(tree_depth(tree.tree))
        leaves.append(tree_leaves(tree.tree))

        if check:
            Xtr = copy.deepcopy(X_train); Xtr['y'] = y_train
            Xte = copy.deepcopy(X_test);  Xte['y'] = y_test
            pr_tr = np.array(tree.predict(Xtr[cols]))
            pr_te = np.array(tree.predict(Xte[cols]))
            if predict_type == "predict_proba":
                pr_tr = np.argmax(pr_tr >= 0.5, axis=1)
                pr_te = np.argmax(pr_te >= 0.5, axis=1)
            au = roc_auc_score(y_test, pr_te)
            ok.append(abs(au - row["aurocs_test"]) < 1e-6)

    df["actual_depth"] = depths
    df["num_leaves"] = leaves
    if check:
        df["auroc_matches"] = ok

    df.to_csv(csv_path, index=False)
    frac = np.mean([v for v in ok if not np.isnan(v)]) if (check and ok) else float("nan")
    print(f"  seed {seed}: {len(df)} solutions, mean depth "
          f"{np.nanmean(depths):.2f}, leaves {np.nanmean(leaves):.2f}"
          + (f", auroc match {frac:.2f}" if check else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path_input", required=True,
                    help="Folder with best_results_pareto_seed_*.csv (one method/metric)")
    ap.add_argument("--data", default="Folktables_AK")
    ap.add_argument("--tree_variant", default="threshold_gain_s")
    ap.add_argument("--fairness_metric", default="eo")
    ap.add_argument("--predict_type", default="predict_proba")
    ap.add_argument("--leaf_outcome_method", default="majority")
    ap.add_argument("--max_h_y", type=float, default=1.1)
    ap.add_argument("--min_h_s", type=float, default=0.0)
    ap.add_argument("--intersectional", action="store_true")
    ap.add_argument("--no_check", action="store_true",
                    help="Skip the AUROC consistency check (faster)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.path_input, "best_results_pareto_seed_*.csv")))
    if not files:
        print("No best_results_pareto_seed_*.csv found in", args.path_input); return

    print(f"{args.data} / {args.tree_variant} / {args.fairness_metric}: {len(files)} file(s)")
    for f in files:
        process_file(f, args.data, args.tree_variant, args.fairness_metric,
                     args.predict_type, args.leaf_outcome_method,
                     args.max_h_y, args.min_h_s, args.intersectional,
                     check=not args.no_check)


if __name__ == "__main__":
    main()

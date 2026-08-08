"""
Recompute actual tree depths for the TWO-TREE runs.

Columns written per row:
  perf_depth, fair_depth, meta_depth,
  max_depth_base = max(perf_depth, fair_depth),
  max_depth_all  = max(perf_depth, fair_depth, meta_depth),
  auroc_matches  (safety check vs. stored aurocs_test)

Folder name encodes: <combination>_AND_<fair_variant>_AND_<perf_variant>
Filename may contain a stray ")" before .csv (from the opt script's format string).

Usage:
  python recompute_depth_two_tree.py --path_input <folder> --data Adult
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
import method as M   # reuse create_performance_tree/create_fair_tree/gamma_sweep helpers


def tree_depth(node):
    if node is None or node.is_leaf:
        return 0
    return 1 + max(tree_depth(c) for c in node.children)


# seed / min_samples_pct / max_depth; tolerate a stray ")" before .csv
FNAME = re.compile(r"best_results_seed_(\d+)_min_samples_(\d+)_max_depth_(\d+)")


def parse_name(fname):
    m = FNAME.search(os.path.basename(fname))
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def parse_folder(path_input):
    """<combination>_AND_<fair_variant>_AND_<perf_variant> -> tuple."""
    # the variant folder is two levels up from the metric folder:
    # .../tradeoffs_two_tree/<combo>_AND_<fair>_AND_<perf>/<metric>/
    parts = os.path.normpath(path_input).split(os.sep)
    for p in parts:
        if "_AND_" in p:
            a = p.split("_AND_")
            if len(a) == 3:
                return a[0], a[1], a[2]   # combination, fair_variant, perf_variant
    raise ValueError("could not parse <combo>_AND_<fair>_AND_<perf> from path")


def build_base_trees(X_train, y_train, s_train, X_test, y_test,
                     perf_variant, fair_variant, split_criterion,
                     max_depth, min_samples, max_h_y, min_h_s,
                     unpriv, pos, predict_type="predict_proba",
                     leaf_outcome_method="majority"):
    """
    Rebuild performance and fair tree AND return the fitted FairClassificationTree
    objects (for depth) alongside the predictions (for the meta tree + check).
    Mirrors create_performance_tree('own') / create_fair_tree('fairness_gain').
    """
    cols = X_train.columns.tolist()
    lom = "probability" if predict_type == "predict_proba" else leaf_outcome_method
    sens = np.array(s_train.values).flatten().tolist()

    Xtr = copy.deepcopy(X_train); Xtr['y'] = y_train
    Xte = copy.deepcopy(X_test);  Xte['y'] = y_test

    # performance tree (perf_variant == "own")
    y_tree = FairClassificationTree(data=Xtr, attributes=cols, idx_target=-1,
                                    unprivileged_group=unpriv, pos_outcome=pos,
                                    threshold_binning=10, sensitive=sens,
                                    leaf_outcome=lom, split_criterion="information_gain")
    y_tree.fit(max_depth=max_depth, min_samples_leave=min_samples,
               tree_type="performance", max_h_y=max_h_y)
    y_preds_train = np.array(y_tree.predict(Xtr[cols]))
    y_preds_test = np.array(y_tree.predict(Xte[cols]))

    # fair tree (fair_variant == "fairness_gain")
    fair_tree = FairClassificationTree(data=Xtr, attributes=cols, idx_target=-1,
                                       unprivileged_group=unpriv, pos_outcome=pos,
                                       threshold_binning=10, sensitive=sens,
                                       leaf_outcome=lom, split_criterion=split_criterion)
    fair_tree.fit(max_depth=max_depth, min_samples_leave=min_samples,
                  tree_type="fair", min_h_s=min_h_s)
    fair_preds_train = np.array(fair_tree.predict(Xtr[cols]))
    fair_preds_test = np.array(fair_tree.predict(Xte[cols]))

    return (y_tree, fair_tree, y_preds_train, y_preds_test,
            fair_preds_train, fair_preds_test, cols)


def process_file(csv_path, data, combination, fair_variant, perf_variant,
                 split_criterion, predict_type, leaf_outcome_method,
                 max_h_y, min_h_s, max_depth_meta, intersectional, check):
    parsed = parse_name(csv_path)
    if parsed is None:
        print("  skip (name not parseable):", os.path.basename(csv_path)); return
    seed, min_samples_pct, max_depth = parsed

    df = pd.read_csv(csv_path)

    X, y, s, unpriv, pos = data_loader_router(data, intersectional)
    X = X.applymap(lambda v: int(v) if isinstance(v, bool) else v).astype(np.float32)
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, pd.DataFrame(s), test_size=0.33, random_state=seed)
    min_samples = int(len(X_train) * min_samples_pct / 100)

    (y_tree, fair_tree, ypr_tr, ypr_te, fpr_tr, fpr_te, cols) = build_base_trees(
        X_train, y_train, s_train, X_test, y_test, perf_variant, fair_variant,
        split_criterion, max_depth, min_samples, max_h_y, min_h_s, unpriv, pos,
        predict_type, leaf_outcome_method)

    perf_d = tree_depth(y_tree.tree)
    fair_d = tree_depth(fair_tree.tree)

    need_meta = combination in ("meta_tree", "meta_tree_optimization")

    # Meta-tree training data: deterministic 50/50 split (random_state=42), exactly as
    # in method.gamma_sweep. Built once; the meta tree itself is (re)fit per gamma.
    meta_val = meta_cols = s_val_split = y_val_split = auroc_optimal = spd_optimal = None
    if need_meta:
        meta_X_train = pd.DataFrame({"y_preds": [p[0] for p in ypr_tr],
                                     "fair_preds": [p[0] for p in fpr_tr], "y": y_train})
        _, meta_val, _, s_val_split, _, y_val_split = train_test_split(
            meta_X_train, s_train, y_train, test_size=0.5, random_state=42,
            stratify=meta_X_train["y"])
        meta_cols = ["y_preds", "fair_preds"]
        from method import statistical_parity_diff
        auroc_optimal = roc_auc_score(y_train, np.argmax(ypr_tr >= 0.5, axis=1))
        spd_optimal = statistical_parity_diff(np.argmax(fpr_tr >= 0.5, axis=1),
                                              s_train.squeeze().to_numpy(),
                                              unprivileged_group=unpriv, pos_outcome=pos)

    perf_col, fair_col, meta_col = [], [], []
    base_max, all_max, ok = [], [], []

    for _, row in df.iterrows():
        gamma = row["gammas"]
        meta_d = np.nan
        meta_tree = None

        if need_meta:
            sens_val = np.array(s_val_split.values).flatten().tolist()
            if combination == "meta_tree_optimization":
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)

                def objective(trial):
                    md = trial.suggest_int("max_depth", 1, 10)
                    ms = trial.suggest_int("min_samples", 2, int(len(y_val_split) / 4))
                    m = FairClassificationTree(data=meta_val, attributes=meta_cols, idx_target=-1,
                                               unprivileged_group=unpriv, pos_outcome=pos,
                                               threshold_binning=10, sensitive=sens_val,
                                               leaf_outcome="probability",
                                               split_criterion="chebyshev", gamma=gamma)
                    m.fit(max_depth=md, min_samples_leave=ms, tree_type="meta_tree_optimization")
                    yv = np.argmax(np.array(m.predict(meta_val)) >= 0.5, axis=1)
                    spd_v = statistical_parity_diff(yv, np.asarray(s_val_split), unpriv, pos)
                    au_v = roc_auc_score(y_val_split, yv)
                    return max((1 - gamma) * abs(auroc_optimal - au_v),
                               gamma * abs(spd_optimal - spd_v))

                study = optuna.create_study(direction="minimize",
                                            sampler=optuna.samplers.TPESampler())
                study.optimize(objective, n_trials=30, n_jobs=1)
                meta_tree = FairClassificationTree(data=meta_val, attributes=meta_cols, idx_target=-1,
                                                   unprivileged_group=unpriv, pos_outcome=pos,
                                                   threshold_binning=10, sensitive=sens_val,
                                                   leaf_outcome="probability",
                                                   split_criterion="chebyshev", gamma=gamma)
                meta_tree.fit(max_depth=study.best_trial.params["max_depth"],
                              min_samples_leave=study.best_trial.params["min_samples"])
            else:  # meta_tree
                meta_tree = FairClassificationTree(data=meta_val, attributes=meta_cols, idx_target=-1,
                                                   unprivileged_group=unpriv, pos_outcome=pos,
                                                   threshold_binning=10, sensitive=sens_val,
                                                   leaf_outcome="probability",
                                                   split_criterion="chebyshev", gamma=gamma)
                meta_tree.fit(max_depth=max_depth, min_samples_leave=min_samples,
                              tree_type="meta_tree")
            meta_d = tree_depth(meta_tree.tree) if meta_tree.tree is not None else np.nan
            preds_test = M.combined_predict_proba(ypr_te, fpr_te, gamma, meta_tree=meta_tree)
        else:
            preds_test = M.combined_predict_proba(ypr_te, fpr_te, gamma)

        perf_col.append(perf_d); fair_col.append(fair_d); meta_col.append(meta_d)
        base_max.append(max(perf_d, fair_d))
        all_max.append(np.nanmax([perf_d, fair_d, meta_d]))

        if check:
            au = roc_auc_score(y_test, preds_test)
            ok.append(bool(abs(au - row["aurocs_test"]) < 1e-6))

    df["perf_depth"] = perf_col
    df["fair_depth"] = fair_col
    df["meta_depth"] = meta_col
    df["max_depth_base"] = base_max
    df["max_depth_all"] = all_max
    if check:
        df["auroc_matches"] = ok
    df.to_csv(csv_path, index=False)

    frac = np.mean([v for v in ok if not (isinstance(v, float) and np.isnan(v))]) if (check and ok) else float("nan")
    print(f"  seed {seed} [{combination}]: perf {perf_d}, fair {fair_d}, "
          f"mean meta {np.nanmean(meta_col):.2f}"
          + (f", auroc match {frac:.2f}" if check else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path_input", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split_criterion", default="gain_s")
    ap.add_argument("--predict_type", default="predict_proba")
    ap.add_argument("--leaf_outcome_method", default="majority")
    ap.add_argument("--max_h_y", type=float, default=1.1)
    ap.add_argument("--min_h_s", type=float, default=0.0)
    ap.add_argument("--max_depth_meta", type=int, default=10)
    ap.add_argument("--intersectional", action="store_true")
    ap.add_argument("--no_check", action="store_true")
    args = ap.parse_args()

    combination, fair_variant, perf_variant = parse_folder(args.path_input)
    files = sorted(glob.glob(os.path.join(args.path_input, "best_results_seed_*.csv")))
    if not files:
        print("No best_results_seed_*.csv in", args.path_input); return
    print(f"{args.data} / {combination}_AND_{fair_variant}_AND_{perf_variant}: {len(files)} file(s)")
    for f in files:
        process_file(f, args.data, combination, fair_variant, perf_variant,
                     args.split_criterion, args.predict_type, args.leaf_outcome_method,
                     args.max_h_y, args.min_h_s, args.max_depth_meta,
                     args.intersectional, check=not args.no_check)


if __name__ == "__main__":
    main()

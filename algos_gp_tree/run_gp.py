"""
Runner for evolutionary decision-tree induction (structural MOO).

Runs NSGA-II or SMS-EMOA over decision-tree topology and writes the
resulting Pareto front in the schema the existing evaluation pipeline consumes
(``best_results_pareto_seed_<N>.csv`` with ``aurocs_test`` / ``spds_test``).

Example:
    ./venv/bin/python algos_gp_tree/run_gp.py --data Compas --algo nsga2 \
        --p_size 40 --num_gen 30 --max_depth 6 --seed 1
"""
import sys
import os
import time
import argparse
import warnings

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'FairTree'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'algos_two_trees'))
sys.path.append('../algos_two_trees')

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, accuracy_score
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.termination.collection import TerminationCollection

from constants import PROJECT_ROOT
from get_data import data_loader_router

import genome as g
import predict as pr
from problem import GPTreeProblem, compute_fairness
from operators import (TreeSampling, TreeCrossover, TreeMutation,
                       TreeDuplicateElimination)


def build_algorithm(name, p_size, meta, usable, max_depth, rng, feat_p=None):
    sampling = TreeSampling(meta, usable, max_depth, rng, feat_p=feat_p)
    crossover = TreeCrossover(max_depth, rng)
    mutation = TreeMutation(meta, usable, max_depth, rng, feat_p=feat_p)
    dup = TreeDuplicateElimination()
    common = dict(pop_size=p_size, sampling=sampling, crossover=crossover,
                  mutation=mutation, eliminate_duplicates=dup)
    if name == "nsga2":
        return NSGA2(**common)
    elif name == "sms":
        return SMSEMOA(**common)
    raise ValueError("unknown algo: %s" % name)


def main():
    parser = argparse.ArgumentParser(description="Evolutionary DT induction (structural MOO).")
    parser.add_argument('--data', type=str, default='Dutch',
                        help='Compas, Adult, Banks, German, Law, Dutch, Folktables_AK, Folktables_HI')
    parser.add_argument('--algo', type=str, default='sms', choices=['nsga2', 'sms'])
    parser.add_argument('--results_path', type=str, default='default')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_cv', type=int, default=3)
    parser.add_argument('--p_size', type=int, default=50)
    parser.add_argument('--num_gen', type=int, default=50)
    parser.add_argument('--max_depth', type=int, default=13)
    parser.add_argument('--max_thresholds', type=int, default=32,
                        help='Candidate thresholds per continuous feature: upper bound when '
                             '--binning midpoints, exact count when --binning uniform')
    parser.add_argument('--binning', type=str, default='midpoints',
                        choices=['midpoints', 'uniform'],
                        help='Threshold placement for continuous features: "midpoints" '
                             '(data-adaptive, default) or "uniform" (evenly spaced across the '
                             'value range, mirrors a fixed-bin greedy scheme). With "uniform" '
                             '--max_thresholds becomes an exact per-feature count.')
    parser.add_argument('--max_time', type=str, default='60:00:00')
    parser.add_argument('--intersectional', action='store_true')
    parser.add_argument('--fair_ig_bias', action='store_true',
                        help='Bias feature selection in sampling/mutation by fair information gain')
    parser.add_argument('--parsimony', type=float, default=0.0,
                        help='Soft bloat penalty coefficient added to the performance objective')
    parser.add_argument('--fairness_metric', type=str, default='accuracy_diff',
                        choices=['spd', 'equalized_odds', 'accuracy_diff'],
                        help='Second objective to minimise: "spd", "equalized_odds" or '
                             '"accuracy_diff". Note: with --intersectional only "spd" is supported.')
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    if args.intersectional and args.fairness_metric != "spd":
        raise ValueError('--intersectional only supports --fairness_metric "spd"')

    date = time.strftime("%Y-%m-%d")
    print("Data=%s algo=%s seed=%d pop=%d gen=%d max_depth=%d"
          % (args.data, args.algo, args.seed, args.p_size, args.num_gen, args.max_depth))

    # load + split (identical protocol to the existing pymoo scripts)
    X, y, s, unprivileged_group, pos_outcome = data_loader_router(args.data, args.intersectional)
    X = X.applymap(lambda v: int(v) if isinstance(v, bool) else v).astype(np.float32)
    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
        X, y, pd.DataFrame(s), test_size=0.33, random_state=args.seed)

    X_train = X_tr.to_numpy(np.float64)
    X_test = X_te.to_numpy(np.float64)
    y_train = y_tr.to_numpy().astype(np.int64).ravel()
    y_test = y_te.to_numpy().astype(np.int64).ravel()
    s_train = s_tr.to_numpy().ravel()
    s_test = s_te.to_numpy().ravel()
    num_classes = int(y_train.max()) + 1

    folds = [(np.asarray(fit_i), np.asarray(val_i))
             for fit_i, val_i in KFold(n_splits=args.num_cv).split(X_train)]

    meta, usable = g.build_feature_meta(X_train, max_thresholds=args.max_thresholds,
                                        rng=np.random.default_rng(args.seed),
                                        binning=args.binning)
    print("features usable for splits: %d / %d" % (len(usable), X_train.shape[1]))

    feat_p = None
    if args.fair_ig_bias:
        from fair_ig import fair_ig_feature_weights
        feat_p = fair_ig_feature_weights(X_train, y_train, s_train, meta, usable)
        print("fair-IG feature bias enabled (top weight=%.3f)" % feat_p.max())

    problem = GPTreeProblem(folds, X_train, y_train, s_train, num_classes,
                            unprivileged_group, pos_outcome, intersectional=args.intersectional,
                            parsimony=args.parsimony, fairness_metric=args.fairness_metric)

    rng = np.random.default_rng(args.seed)
    algorithm = build_algorithm(args.algo, args.p_size, meta, usable, args.max_depth, rng,
                                feat_p=feat_p)
    termination = TerminationCollection(
        get_termination("n_gen", args.num_gen),
        get_termination("time", args.max_time),
    )

    t0 = time.time()
    res = minimize(problem, algorithm, termination, seed=args.seed, verbose=True)
    wall = time.time() - t0

    # normalise res.X to a list of trees
    sols = res.X
    if sols is None:
        print("No solutions returned.")
        return
    if sols.ndim == 1:
        sols = sols.reshape(1, -1)

    # retrain leaves on full train, evaluate on held-out test
    rows = []
    for i, row in enumerate(sols):
        tree = row[0]
        pr.fit_leaves(tree, X_train, y_train, num_classes, fallback=problem.global_major)
        preds_train = pr.predict(tree, X_train)
        preds_test = pr.predict(tree, X_test)
        rows.append({
            "solution_id": i,
            "aurocs_train": roc_auc_score(y_train, preds_train),
            "aurocs_test": roc_auc_score(y_test, preds_test),
            "accs_train": accuracy_score(y_train, preds_train),
            "accs_test": accuracy_score(y_test, preds_test),
            "spds_train": compute_fairness(preds_train, y_train, s_train, unprivileged_group,
                                           pos_outcome, args.fairness_metric, args.intersectional),
            "spds_test": compute_fairness(preds_test, y_test, s_test, unprivileged_group,
                                          pos_outcome, args.fairness_metric, args.intersectional),
            "depth": g.depth(tree),
            "size": g.size(tree),
        })
    results_df = pd.DataFrame(rows)

    base = PROJECT_ROOT if args.results_path == "default" else args.results_path
    out_dir = os.path.join(base, "results_best_gp_" + args.algo + "_" + args.fairness_metric + "_" + date, str(args.data),
                           "tradeoffs_gp_tree", args.algo)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "best_results_pareto_seed_%d.csv" % args.seed)
    results_df.to_csv(out_csv, index=False)

    # budget log (for the fairness-of-comparison controls)
    n_eval = getattr(getattr(res, "algorithm", None), "evaluator", None)
    n_eval = getattr(n_eval, "n_eval", args.p_size * args.num_gen)
    meta_df = pd.DataFrame([{
        "algo": args.algo, "data": args.data, "seed": args.seed,
        "pop_size": args.p_size, "n_gen": args.num_gen, "max_depth": args.max_depth,
        "binning": args.binning, "max_thresholds": args.max_thresholds,
        "fairness_metric": args.fairness_metric,
        "fair_ig_bias": args.fair_ig_bias, "parsimony": args.parsimony,
        "n_eval": n_eval, "wall_clock_s": round(wall, 2), "front_size": len(rows),
    }])
    meta_df.to_csv(os.path.join(out_dir, "run_meta_seed_%d.csv" % args.seed), index=False)

    print("\nFront (%d solutions) in %.1fs, %s model-evaluations:" % (len(rows), wall, n_eval))
    print(results_df[["aurocs_test", "spds_test", "depth", "size"]]
          .sort_values("spds_test").to_string(index=False))
    print("\nSaved:", out_csv)


if __name__ == '__main__':
    main()

import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
import uuid
from pymoo.termination import get_termination
from pymoo.termination.collection import TerminationCollection

sys.path.append('../algos_two_trees')
sys.path.append('../algos_one_tree')
sys.path.append('../FairTree')
sys.path.append('../')

import warnings
import time
import copy
import numpy as np
import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split, KFold
from get_data import data_loader_router
from sklearn.metrics import roc_auc_score, accuracy_score
from utils import statistical_parity_diff
from fairlearn.metrics import demographic_parity_difference
from constants import PROJECT_ROOT
from fair_classification_tree import FairClassificationTree
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination
from algos_one_tree.method import create_tree


class FairnessAurocProblem(ElementwiseProblem):
    """
    Defines a bi-objective optimization problem with the pymoo APIs.
    Here, the "design variables" are the hyperparameters we consider: gamma,
    min samples and max depth.
    """

    def __init__(self, folds, train_data, pos_outcome=1, unprivileged_group='0',
                 upper_min_samples=50, upper_max_depth=13,
                 variant='chebyshev', args=dict(), current_time='25-00-00'):
        self.variant = variant
        self.kf = folds
        self.train_data = train_data
        self.pos_outcome = pos_outcome
        self.unprivileged_group = unprivileged_group
        self.args = args
        self.time = current_time

        if variant == "threshold_gain_s" or variant == "backtracking":
            variables = {
                "gamma": Real(bounds=(0.0, 0.2)),
                "min_samples": Integer(bounds=(1, upper_min_samples)),
                "max_depth": Integer(bounds=(1, upper_max_depth)),
            }
        elif variant == "weighted_combi" or variant == "chebyshev":
            variables = {
                "gamma": Real(bounds=(0.0, 1.0)),
                "min_samples": Integer(bounds=(1, upper_min_samples)),
                "max_depth": Integer(bounds=(1, upper_max_depth)),
            }
        else:
            raise ValueError("given variant for training the tree is unknown")
        super().__init__(vars=variables, n_obj=2)

    def _evaluate(self, x, out, *args, **kwargs):
        print(x)
        gamma, min_samples, max_depth = x
        eval_id = uuid.uuid4().hex
        X_train, y_train, s_train = self.train_data
        k = 0

        if self.args.results_path == "default":
            directory = os.path.join(PROJECT_ROOT, 'results_cv_pymoo_' + self.time, str(self.args.data),
                                     "tradeoffs_one_tree", str(self.args.tree_variant), "seed_" + str(self.args.seed),
                                     "min_samples_" + str(x['min_samples']), "max_depth_" + str(x['max_depth'])
                                                                                                + "_evalid_" + str(eval_id))
        else:
            directory = os.path.join(self.args.results_path, 'results_cv_pymoo_' + self.time, str(self.args.data),
                                     "tradeoffs_one_tree", str(self.args.tree_variant), "seed_" + str(self.args.seed),
                                     "min_samples_" + str(x['min_samples']), "max_depth_" + str(x['max_depth'])
                                                                                                + "_evalid_" + str(eval_id))

        min_samples = int(len(X_train) * x['min_samples'] / 100)  # this is handled last-minute so we still know which are the good script args
        results = []
        for train_index, val_index in self.kf.split(X_train):
            X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]
            s_train_cv, s_val_cv = s_train.iloc[train_index], s_train.iloc[val_index]

            results_kf = create_trees(X_train_cv, X_val_cv, y_train_cv, y_val_cv, s_train_cv, s_val_cv, self.unprivileged_group, self.pos_outcome,
                                      self.args.predict_type, self.args.data, self.args.tree_variant, self.args.leaf_outcome_method, x['max_depth'],
                                      min_samples, self.args.max_h_y, self.args.min_h_s, x['gamma'], k, directory,
                                      intersectional=self.args.intersectional)
            results.append(results_kf)
            k += 1

        # concatenate results from different folds
        combined_results = defaultdict(list)
        for d in results:
            for k, v in d.items():
                combined_results[k].extend(v)

        results_df = pd.DataFrame(combined_results)

        # zip predictions folder and remove original one to have less saved files
        # pred_dir = Path(directory) / "predictions"
        # tar_path = pred_dir.with_suffix(".tar.gz")
        # if pred_dir.exists():
        #     subprocess.run(
        #         ["tar", "-czf", tar_path, "-C", directory, "predictions"],
        #         check=True,
        #     )
        #     shutil.rmtree(pred_dir)

        auroc = 1 - results_df['aurocs_test'].mean()  # 1-auroc because we want to minimize
        fair = results_df['spds_test'].mean()
        out['F'] = [auroc, fair]


def main():
    parser = argparse.ArgumentParser(description='Get run values.')

    parser.add_argument('--data', type=str, default='Dutch',
                        help='Dataset to use: Compas, Adult, Banks, German, Law, Dutch')

    parser.add_argument('--results_path', type=str, default='default',
                        help='"default" or custom path')

    parser.add_argument('--seed', type=int, default='1',
                        help='Seed for holdout')

    parser.add_argument('--num_cv', type=int, default='3',
                        help='Number of folds for cross-validation')

    parser.add_argument('--predict_type', type=str, default='predict_proba',
                        help='Type of prediction, has to be in ["predict", "predict_proba"]')

    parser.add_argument('--tree_variant', type=str, default='weighted_combi',
                        help='Type of tree that will be used as "fair" tree, has to be in '
                             '["threshold_gain_s", "weighted_combi", "backtracking", "chebyshev"]')

    parser.add_argument('--leaf_outcome_method', type=str, default='majority',
                        help='Which value or model is saved in the leaves to give a tree output, has to be in '
                             '["mean", "majority"]')

    parser.add_argument('--max_depth', type=int, default='13',
                        help='Maximum depth of optimal tree (for deeper trees runtime increases). Note that this is '
                             'actually the maximum max depth taken\
                              by pymoo.')

    parser.add_argument('--min_samples', type=int, default='25',
                        help='Minimum samples per leaf. This is a percentage. Note that this is actually the maximum '
                             'min samples (upper bound) taken by pymoo.')

    parser.add_argument('--max_h_y', type=float, default='1.1',
                        help="Maximum entropy/uncertainty that should be there regarding y in a leaf of the performance"
                             "tree")

    parser.add_argument('--min_h_s', type=float, default='0.0',
                        help="Minimum entropy/uncertainty that should be there regarding s in a leaf of the fair"
                             "tree")

    parser.add_argument('--num_gammas', type=int, default='50',
                        help='Number of gamma values between 0 and 1')

    parser.add_argument('--p_size', type=int, default='50',
                        help='Size of population')

    parser.add_argument('--num_gen', type=int, default='50',
                        help='Number of generations')

    parser.add_argument('--max_time', type=str, default='20:00:00',
                        help='Maximum time until termination of evolutionary algorithm')

    parser.add_argument('--intersectional', action='store_true',
                        help='Sets up intersectional sensitive attributes and calculation')

    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    current_time = time.strftime("%Y-%m-%d")

    print("Current time:", current_time)

    X, y, s, unprivileged_group, pos_outcome = data_loader_router(args.data, args.intersectional)

    X = X.applymap(lambda x: int(x) if isinstance(x, bool) else x).astype(np.float32)

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, pd.DataFrame(s), test_size=0.33, random_state=args.seed
    )

    kf = KFold(n_splits=args.num_cv)

    train_data = (X_train, y_train, s_train)
    test_data = (X_test, y_test, s_test)

    problem = FairnessAurocProblem(kf, train_data, upper_min_samples=args.min_samples, upper_max_depth=args.max_depth,
                                   pos_outcome=pos_outcome, unprivileged_group=unprivileged_group,
                                   variant=args.tree_variant, args=args, current_time=current_time)

    # Terminate when ANY condition is met
    termination = TerminationCollection(
        get_termination("n_gen", args.num_gen),  # Max generations
        get_termination("time", args.max_time)  # Max hours
    )
    algo = NSGA2(
        pop_size=args.p_size,
        sampling=MixedVariableSampling(),
        mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
        eliminate_duplicates=False,
        verbose=False
    )
    res = minimize(problem, algo, termination, seed=args.seed, verbose=False)

    print(res.F)
    print(res.X)
    for i, (x, f) in enumerate(zip(res.X, res.F)):
        print(f"Solution {i}:")
        print(f"  gamma={x['gamma']}, min_samples={x['min_samples']}, max_depth={x['max_depth']}")
        print(f"  auroc={1-f[0]}, unfairness={f[1]}")
    print(res)

    # for each solution (hp combination), retrain the model and evaluate on test data and save all these to results dict

    # create results_best folder corresponding to optimized population (solutions)
    # save in the same structure the solutions, i.e. one file for this run that is named according to the seed
    # -> csv file should contain all solutions (=trade off points of curve), one per row, with auroc, acc, spd
    # (train/test) of the results dict. Also the solution should contain the hp combination for this solution

    # -> afterwards, we can evaluate the 15 files (or 1 file, depending on how many runs of EA we do)

    # retrain each Pareto-optimal solution and evaluate on train and test set
    final_results = []

    if args.results_path == "default":
        results_best_dir = os.path.join(
            PROJECT_ROOT,
            "results_best_pymoo_" + current_time,
            str(args.data),
            "tradeoffs_one_tree",
            str(args.tree_variant))
    else:
        results_best_dir = os.path.join(
            args.results_path,
            "results_best_pymoo_" + current_time,
            str(args.data),
            "tradeoffs_one_tree",
            str(args.tree_variant))

    os.makedirs(results_best_dir, exist_ok=True)

    for i, x in enumerate(res.X):
        gamma = x["gamma"]
        max_depth = x["max_depth"]
        min_samples_pct = x["min_samples"]
        min_samples = int(len(X_train) * min_samples_pct / 100)

        # train tree on full train set, evaluate on test
        results = create_trees(
            X_train, X_test, y_train, y_test, s_train, s_test,
            unprivileged_group,
            pos_outcome,
            args.predict_type,
            args.data,
            args.tree_variant,
            args.leaf_outcome_method,
            max_depth,
            min_samples,
            args.max_h_y,
            args.min_h_s,
            gamma,
            k_cv="full",  # distinguish from CV folds
            directory=None,
            intersectional=args.intersectional,
            save_preds=False
        )

        # results dict has lists → extract scalars
        row = {
            "solution_id": i,
            "gamma": gamma,
            "min_samples_pct": min_samples_pct,
            "min_samples_abs": min_samples,
            "max_depth": max_depth,

            "aurocs_train": results["aurocs_train"][0],
            "aurocs_test": results["aurocs_test"][0],
            "accs_train": results["accs_train"][0],
            "accs_test": results["accs_test"][0],
            "spds_train": results["spds_train"][0],
            "spds_test": results["spds_test"][0],
        }

        final_results.append(row)

    # Save final results
    results_df = pd.DataFrame(final_results)
    results_csv_path = os.path.join(results_best_dir, f"best_results_pareto_seed_{args.seed}.csv")
    results_df.to_csv(results_csv_path, index=False)

    print("\nFinal Pareto evaluation saved to:")
    print(results_csv_path)


def create_trees(X_train, X_test, y_train, y_test, s_train, s_test, unprivileged_group, pos_outcome, predict_type,
                 data, tree_variant, leaf_outcome_method, max_depth, min_samples, max_h_y, min_h_s, gamma, k_cv,
                 directory, intersectional=False, save_preds=False):
    results = {
        "spds_train": [],
        "spds_test": [],
        "aurocs_train": [],
        "aurocs_test": [],
        "accs_train": [],
        "accs_test": []
    }

    # create tree and get its predictions
    preds_train, preds_test = create_tree(X_train, y_train, s_train, X_test, y_test, tree_variant, predict_type,
                                          max_depth, min_samples, max_h_y, min_h_s, leaf_outcome_method,
                                          unprivileged_group, pos_outcome,
                                          gamma, print_trees=False)

    if save_preds:
        os.makedirs(os.path.join(directory, 'predictions'), exist_ok=True)

        if preds_train is not None:
            pd.DataFrame({'preds_train': preds_train}).to_csv(
                os.path.join(directory, 'predictions', 'preds_train_fold{}_gamma{}_max_depth_{}_min_samples_{}.csv'.format(k_cv, gamma, max_depth, min_samples)))
        else:
            pd.DataFrame({'preds_train': [preds_train]}).to_csv(
                os.path.join(directory, 'predictions', 'preds_train_fold{}_gamma{}_max_depth_{}_min_samples_{}.csv'.format(k_cv, gamma, max_depth, min_samples)))

        if preds_test is not None:
            pd.DataFrame({'preds_test': preds_test}).to_csv(
                os.path.join(directory, 'predictions', 'preds_test_fold{}_gamma{}_max_depth_{}_min_samples_{}.csv'.format(k_cv, gamma, max_depth, min_samples)))
        else:
            pd.DataFrame({'preds_test': [preds_test]}).to_csv(
                os.path.join(directory, 'predictions', 'preds_test_fold{}_gamma{}_max_depth_{}_min_samples_{}.csv'.format(k_cv, gamma, max_depth, min_samples)))

    if preds_train is None:
        # empty tree
        results["spds_train"].append(-1)
        results["spds_test"].append(-1)
        results["aurocs_train"].append(-1)
        results["aurocs_test"].append(-1)
        results["accs_train"].append(-1)
        results["accs_test"].append(-1)
    else:
        if intersectional:
            spd_train = demographic_parity_difference(y_train, preds_train, sensitive_features=s_train)
            spd_test = demographic_parity_difference(y_test, preds_test, sensitive_features=s_test)
        else:
            spd_train = statistical_parity_diff(preds_train, np.asarray(s_train), unprivileged_group, pos_outcome)
            spd_test = statistical_parity_diff(preds_test, np.asarray(s_test), unprivileged_group, pos_outcome)
        auroc_train = roc_auc_score(y_train, preds_train)
        auroc_test = roc_auc_score(y_test, preds_test)
        acc_train = accuracy_score(y_train, preds_train)
        acc_test = accuracy_score(y_test, preds_test)

        results["spds_train"].append(spd_train)
        results["spds_test"].append(spd_test)
        results["aurocs_train"].append(auroc_train)
        results["aurocs_test"].append(auroc_test)
        results["accs_train"].append(acc_train)
        results["accs_test"].append(acc_test)
    
    # save results to csv
    if save_preds:
        pd.DataFrame(results).to_csv(
            os.path.join(directory, 'results_fold{}.csv'.format(k_cv)))
    return results


if __name__ == '__main__':
    main()

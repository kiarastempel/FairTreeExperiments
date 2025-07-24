import numpy as np
import argparse
import datetime

import pandas as pd

from algos_two_trees.get_data import *
from algos_two_trees.method import *
from constants import PROJECT_ROOT
import sys
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from evaluation.evaluation_metrics import autoc, num_unique_points, num_local_pareto_points, check_point_distribution


seed = 42


def main():

    num_cv = 3
    num_seeds = 15

    parser = argparse.ArgumentParser(description='Get run values.')

    parser.add_argument('--path_input', type=str, default='results_cv_flipped/Compas/tradeoffs_two_tree/fairness_gain_AND_sklearn',
                        help='Path to specific input results folder')

    parser.add_argument('--path_output', type=str,
                        default='results_cp/Compas/tradeoffs_two_trees/fairness_gain_AND_sklearn',
                        help='Path to specific folder where the output will be saved')

    parser.add_argument('--data', type=str, default='Compas',
                        help='Dataset to use: Compas, Adult, Banks, German, Law, Dutch')

    parser.add_argument('--predict_type', type=str, default='predict_proba',
                        help='Type of prediction, has to be in ["predict", "predict_proba"]')

    parser.add_argument('--fair_tree_variant', type=str, default='fairness_gain',
                        help='Type of tree that will be used as "fair" tree, has to be in '
                             '["relabel", "fairness_gain", "optimal_gain_s", "optimal_spd", "random"]')

    parser.add_argument('--performance_tree_variant', type=str, default='sklearn',
                        help='Type of tree that will be used as "fair" tree, has to be in '
                             '["sklearn", "optimal", "own"]')

    parser.add_argument('--split_criterion', type=str, default='gain_s',
                        help='Indicates which method is used in split criterion for the fair tree, has to be in '
                             '["information_gain", "gain_s", "spd", "threshold_constraint"]')

    parser.add_argument('--leaf_outcome_method', type=str, default='majority',
                        help='Which value or model is saved in the leaves to give a tree output, has to be in '
                             '["mean", "majority"]')

    parser.add_argument('--num_gammas', type=int, default='50',
                        help='Number of gamma values between 0 and 1')

    args = parser.parse_args()

    path = os.path.join(PROJECT_ROOT, args.path_input)

    if args.data == "Compas":
        X, y, s, unprivileged_group, pos_outcome = get_compas(os.path.join(PROJECT_ROOT, 'data', 'compas-preprocessed.csv'))
    elif args.data == "Adult":
        X, y, s, unprivileged_group, pos_outcome = get_adult()
    elif args.data == "Banks":
        X, y, s, unprivileged_group, pos_outcome = get_banks()
    elif args.data == "German":
        X, y, s, unprivileged_group, pos_outcome = get_german()
    elif args.data == "Law":
        X, y, s, unprivileged_group, pos_outcome = get_law()
    elif args.data == "Dutch":
        X, y, s, unprivileged_group, pos_outcome = get_dutch_census()
    else:
        raise ValueError("unknown dataset")

    directory = os.path.join(PROJECT_ROOT, args.path_output)
    os.makedirs(directory, exist_ok=True)

    for holdout_seed in range(1, num_seeds + 1):
        X = X.applymap(lambda x: int(x) if isinstance(x, bool) else x).astype(np.float32)
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, pd.DataFrame(s), test_size=0.33, random_state=holdout_seed
        )
        hps = {"min_samples": [], "max_depth": [], "autoc": []}
        print(holdout_seed)
        for min_samples in [25, 10, 1]:
            for max_depth in [4, 6, 8, 13]:
                autoc_scores = []
                for k in range(num_cv):
                    # check auroc/fairness on test data for fold k
                    results = pd.read_csv(os.path.join(path, 'seed_' + str(holdout_seed),
                                                             'min_samples_' + str(min_samples),
                                                             'max_depth_' + str(max_depth),
                                                             'results_fold' + str(k) + '.csv'))

                    aurocs_test = results["aurocs_test"]
                    for i, auroc in enumerate(aurocs_test):
                        if auroc < 0.5:
                            aurocs_test[i] = 1 - aurocs_test[i]
                    # calculate AUTOC for fold and save with hps in dict
                    autoc_score = num_unique_points(results['spds_test'], aurocs_test)
                    autoc_scores.append(autoc_score)
                avg_autoc = np.mean(autoc_scores)
                hps['min_samples'].append(min_samples)
                hps['max_depth'].append(max_depth)
                hps['autoc'].append(avg_autoc)
        # look for best hp combination
        best_autoc = max(hps['autoc'])
        idx_best_autoc = hps['autoc'].index(best_autoc)
        best_hps = {
            "min_samples": hps['min_samples'][idx_best_autoc],
            "max_depth": hps['max_depth'][idx_best_autoc]}

        best_min_samples = int(len(X_train) * best_hps["min_samples"] / 100)
        best_max_depth = best_hps["max_depth"]

        # retrain model using best hps on X_train got by holdout_seed, test on corresponding X_test
        y_preds_train, y_preds_test = create_performance_tree(X_train, y_train, s_train, X_test, y_test,
                                                              args.performance_tree_variant,
                                                              args.predict_type,
                                                              best_max_depth, best_min_samples,
                                                              args.leaf_outcome_method,
                                                              unprivileged_group, pos_outcome, args.split_criterion,
                                                              print_trees=False)
        fair_preds_train, fair_preds_test = create_fair_tree(X_train, y_train, s_train, X_test, y_test,
                                                             args.fair_tree_variant,
                                                             args.predict_type,
                                                             best_max_depth, best_min_samples,
                                                             args.leaf_outcome_method,
                                                             unprivileged_group, pos_outcome, args.split_criterion,
                                                             print_trees=False)

        # append to results_opt
        # from all 15 curves, take for plotting the one with median autoc
        # for table: take average of all 15 best autocs
        results = gamma_sweep(args.predict_type, y_preds_train, y_preds_test, fair_preds_train, fair_preds_test,
                              s_train, s_test, y_train, y_test, unprivileged_group, pos_outcome, args.data,
                              args.fair_tree_variant, args.performance_tree_variant, args.split_criterion,
                              args.num_gammas, timed_run=False, print_trees=False)
        pd.DataFrame(results).to_csv(
            os.path.join(directory, 'best_results_seed_{}_min_samples_{}_max_depth_{}).csv'.format(
                holdout_seed, best_hps["min_samples"], best_hps["max_depth"])))


if __name__ == "__main__":
    main()

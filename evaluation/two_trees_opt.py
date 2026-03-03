import sys
sys.path.append('../evaluation')
sys.path.append('../')

import warnings
import numpy as np
import argparse
import datetime
import pandas as pd
from algos_two_trees.get_data import *
from algos_two_trees.method import *
from constants import PROJECT_ROOT
import sys
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from evaluation_metrics import autoc, hypervolume, num_unique_points, num_local_pareto_points, check_point_distribution


seed = 42


def main():

    num_cv = 3
    num_seeds = 15

    parser = argparse.ArgumentParser(description='Get run values.')

    parser.add_argument('--path_input', type=str, default='results_cv_MLJ_old_opt/Folktables_HI/tradeoffs_two_tree/meta_tree_AND_fairness_gain_AND_sklearn',
                        help='Path to specific input results folder')

    parser.add_argument('--path_output', type=str,
                        default='results_best_MLJ_old_opt_hypervolume/Folktables_HI/tradeoffs_two_tree/meta_tree_AND_fairness_gain_AND_sklearn',
                        help='Path to specific folder where the output will be saved')

    parser.add_argument('--data', type=str, default='Folktables_HI',
                        help='Dataset to use: Compas, Adult, Banks, German, Law, Dutch, Folktables_AK, Folktables_HI')

    parser.add_argument('--predict_type', type=str, default='predict_proba',
                        help='Type of prediction, has to be in ["predict", "predict_proba"]')

    parser.add_argument('--fair_tree_variant', type=str, default='fairness_gain',
                        help='Type of tree that will be used as "fair" tree, has to be in '
                             '["relabel", "fairness_gain", "optimal_gain_s", "optimal_spd", "random"]')

    parser.add_argument('--performance_tree_variant', type=str, default='own',
                        help='Type of tree that will be used as "fair" tree, has to be in '
                             '["sklearn", "optimal", "own"]')

    parser.add_argument('--num_performance_ensemble', type=int, default='1',
                        help='If performance_ensemble is chosen as tree variant for the performance tree: '
                             'number of ensemble members')

    parser.add_argument('--num_fair_ensemble', type=int, default='1',
                        help='If fair_ensemble is chosen as tree variant for the fair tree: number of ensemble members')

    parser.add_argument('--split_criterion', type=str, default='gain_s',
                        help='Indicates which method is used in split criterion for the fair tree, has to be in '
                             '["information_gain", "gain_s", "spd", "threshold_constraint"]')

    parser.add_argument('--leaf_outcome_method', type=str, default='majority',
                        help='Which value or model is saved in the leaves to give a tree output, has to be in '
                             '["mean", "majority"]')

    parser.add_argument('--num_gammas', type=int, default='50',
                        help='Number of gamma values between 0 and 1')

    parser.add_argument('--max_h_y', type=float, default='1.1',
                        help="Maximum entropy/uncertainty that should be there regarding y in a leaf of the performance"
                             "tree")

    parser.add_argument('--min_h_s', type=float, default='0.0',
                        help="Minimum entropy/uncertainty that should be there regarding s in a leaf of the fair")

    parser.add_argument('--intersectional', action='store_true',
                        help='Employ multiple sensitive attributes in data loaders')

    parser.add_argument('--combination', type=str, default='meta_tree',
                        help='How the predictions of the performance tree and the fair tree are combined '
                             '["linear", "meta_tree", "meta_tree_optimization"]')

    parser.add_argument('--optimization_objective', type=str, default='hypervolume',
                        help='Objective to optimize on, "autoc" or "hypervolume"')

    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    path = os.path.join(PROJECT_ROOT, args.path_input)

    X, y, s, unprivileged_group, pos_outcome = data_loader_router(args.data, args.intersectional)

    directory = os.path.join(PROJECT_ROOT, args.path_output)
    os.makedirs(directory, exist_ok=True)

    os.makedirs(os.path.join(PROJECT_ROOT, 'plots_MLJ', 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, 'algo_two_trees_visualization_MLJ'), exist_ok=True)

    for holdout_seed in range(1, num_seeds + 1):
        X = X.applymap(lambda x: int(x) if isinstance(x, bool) else x).astype(np.float32)
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, pd.DataFrame(s), test_size=0.33, random_state=holdout_seed
        )
        hps = {"min_samples": [], "max_depth": [], args.optimization_objective: []}
        print("holdout_seed:", holdout_seed)
        for min_samples in [25, 10, 1]:
            for max_depth in [4, 6, 8, 13]:
                objective_scores = []
                for k in range(num_cv):
                    # check auroc/fairness on test data for fold k

                    file_path = os.path.join(path, 'seed_' + str(holdout_seed),
                                                   'min_samples_' + str(min_samples),
                                                   'max_depth_' + str(max_depth),
                                                   "max_h_y_" + str(args.max_h_y),
                                                   "min_h_s" + str(args.min_h_s),
                                                   'results_fold' + str(k) + '.csv')

                    if not os.path.exists(file_path):
                        continue

                    results = pd.read_csv(file_path)

                    aurocs_test = results["aurocs_test"]
                    for i, auroc in enumerate(aurocs_test):
                        if auroc < 0.5:
                            aurocs_test[i] = 1 - aurocs_test[i]
                    # calculate objective value for fold and save with hps in dict
                    if args.optimization_objective == "autoc":
                        objective_score = autoc(results['spds_test'], aurocs_test)
                    elif args.optimization_objective == "hypervolume":
                        objective_score = hypervolume(results['spds_test'], aurocs_test)
                    objective_scores.append(objective_score)
                avg_objective = np.mean(objective_scores)
                hps['min_samples'].append(min_samples)
                hps['max_depth'].append(max_depth)
                hps[args.optimization_objective].append(avg_objective)
        # look for best hp combination
        best_objective = max(hps[args.optimization_objective])
        idx_best_objective = hps[args.optimization_objective].index(best_objective)
        best_hps = {
            "min_samples": hps['min_samples'][idx_best_objective],
            "max_depth": hps['max_depth'][idx_best_objective]}

        best_min_samples = int(len(X_train) * best_hps["min_samples"] / 100)
        best_max_depth = best_hps["max_depth"]

        # retrain model using best hps on X_train got by holdout_seed, test on corresponding X_test
        y_preds_train, y_preds_test = create_performance_tree(X_train, y_train, s_train, X_test, y_test,
                                                              args.performance_tree_variant,
                                                              args.num_performance_ensemble, #
                                                              args.predict_type,
                                                              best_max_depth, best_min_samples,
                                                              args.max_h_y,
                                                              args.leaf_outcome_method,
                                                              unprivileged_group, pos_outcome, args.split_criterion,
                                                              print_trees=False)
        fair_preds_train, fair_preds_test = create_fair_tree(X_train, y_train, s_train, X_test, y_test,
                                                             args.fair_tree_variant,
                                                             args.num_fair_ensemble,
                                                             args.predict_type,
                                                             best_max_depth, best_min_samples,
                                                             args.min_h_s,
                                                             args.leaf_outcome_method,
                                                             unprivileged_group, pos_outcome, args.split_criterion,
                                                             print_trees=False)

        # append to results_opt
        # from all 15 curves, take for plotting the one with median objective value
        # for table: take average of all 15 best objective values
        results = gamma_sweep(X_train, args.predict_type, y_preds_train, y_preds_test, fair_preds_train, fair_preds_test,
                              s_train, s_test, y_train, y_test, unprivileged_group, pos_outcome, best_max_depth,
                              best_min_samples, args.data,
                              args.fair_tree_variant, args.performance_tree_variant, best_max_depth, args.split_criterion,
                              args.num_gammas, timed_run=False, print_trees=False, intersectional=args.intersectional,
                              combination=args.combination)

        pd.DataFrame(results).to_csv(
            os.path.join(directory, 'best_results_seed_{}_min_samples_{}_max_depth_{}).csv'.format(
                holdout_seed, best_hps["min_samples"], best_hps["max_depth"])))


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import os
import argparse
import copy
from sklearn.model_selection import train_test_split, KFold
from algos_two_trees.get_data import *
from sklearn.metrics import roc_auc_score, accuracy_score
from algos_two_trees.utils import statistical_parity_diff
from constants import PROJECT_ROOT
from algos_two_trees.method import create_performance_tree, create_fair_tree, combined_predict_proba, combined_predict
import warnings


def main():
    parser = argparse.ArgumentParser(description='Get run values.')

    parser.add_argument('--data', type=str, default='Compas',
                        help='Dataset to use: Compas, Adult, Banks, German, Law, Dutch')

    parser.add_argument('--seed', type=int, default='42',
                        help='Seed for holdout')

    parser.add_argument('--num_cv', type=int, default='3',
                        help='Number of folds for cross-validation')

    parser.add_argument('--predict_type', type=str, default='predict_proba',
                        help='Type of prediction, has to be in ["predict", "predict_proba"]')

    parser.add_argument('--fair_tree_variant', type=str, default='fairness_gain',
                        help='Type of tree that will be used as "fair" tree, has to be in '
                             '["fairness_gain", "optimal_gain_s", "optimal_spd"]')

    parser.add_argument('--performance_tree_variant', type=str, default='sklearn',
                        help='Type of tree that will be used as "fair" tree, has to be in '
                             '["sklearn", "optimal"]')

    parser.add_argument('--split_criterion', type=str, default='gain_s',
                        help='Indicates which method is used in split criterion for the fair tree, has to be in '
                             '["information_gain", "gain_s", "spd", "threshold_constraint"]')

    parser.add_argument('--leaf_outcome_method', type=str, default='majority',
                        help='Which value or model is saved in the leaves to give a tree output, has to be in '
                             '["mean", "majority"]')

    parser.add_argument('--max_depth', type=int, default='4',
                        help='Maximum depth of optimal tree (for deeper trees runtime increases)')

    parser.add_argument('--min_samples', type=int, default='25',
                        help='Minimum samples per leave')

    parser.add_argument('--num_gammas', type=int, default='50',
                        help='Number of gamma values between 0 and 1')

    args = parser.parse_args()
    warnings.filterwarnings("ignore")

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

    directory = os.path.join(PROJECT_ROOT, 'results_cv', str(args.data), 'tradeoffs_two_tree',
                             str(args.fair_tree_variant) + "_AND_" + str(args.performance_tree_variant),
                             "seed_" + str(args.seed),
                             "min_samples_" + str(args.min_samples),
                             "max_depth_" + str(args.max_depth))
    os.makedirs(directory, exist_ok=True)

    X = X.applymap(lambda x: int(x) if isinstance(x, bool) else x).astype(np.float32)

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, pd.DataFrame(s), test_size=0.33, random_state=args.seed
    )

    kf = KFold(n_splits=args.num_cv)
    min_samples = int(len(X_train) * args.min_samples / 100)

    k = 0
    for train_index, val_index in kf.split(X_train):
        X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]
        s_train_cv, s_val_cv = s_train.iloc[train_index], s_train.iloc[val_index]

        combine_trees(X_train_cv, X_val_cv, y_train_cv, y_val_cv, s_train_cv, s_val_cv,
                      unprivileged_group, pos_outcome, args.predict_type,
                      args.data, args.fair_tree_variant, args.performance_tree_variant, args.leaf_outcome_method,
                      args.split_criterion, args.max_depth, min_samples, args.num_gammas, k, directory)

        k += 1


def combine_trees(X_train, X_test, y_train, y_test, s_train, s_test, unprivileged_group, pos_outcome, predict_type,
                  data, fair_tree_variant, performance_tree_variant, leaf_outcome_method, split_criterion, max_depth,
                  min_samples, num_gammas, k_cv, directory):
    # get predictions of both separate trees
    y_preds_train, y_preds_test = create_performance_tree(X_train, y_train, s_train, X_test, y_test,
                                                          performance_tree_variant,
                                                          predict_type, max_depth, min_samples, leaf_outcome_method,
                                                          unprivileged_group, pos_outcome, split_criterion,
                                                          print_trees=False)
    fair_preds_train, fair_preds_test = create_fair_tree(X_train, y_train, s_train, X_test, y_test, fair_tree_variant,
                                                         predict_type, max_depth, min_samples, leaf_outcome_method,
                                                         unprivileged_group, pos_outcome, split_criterion,
                                                         print_trees=False)

    # gamma sweep: combine separate predictions using various gamma values
    results = gamma_sweep(predict_type, y_preds_train, y_preds_test, fair_preds_train, fair_preds_test, s_train, s_test,
                          y_train, y_test, unprivileged_group, pos_outcome, num_gammas, k_cv, directory)

    # save results to csv
    pd.DataFrame(results).to_csv(os.path.join(directory, 'results_fold{}.csv'.format(k_cv)))


def gamma_sweep(predict_type, y_preds_train, y_preds_test, fair_preds_train, fair_preds_test, s_train, s_test,
                y_train, y_test, unprivileged_group, pos_outcome, num_gammas, k_cv, directory):
    results = {
        "gammas": [],
        "spds_train": [],
        "spds_test": [],
        "aurocs_train": [],
        "aurocs_test": [],
        "accs_train": [],
        "accs_test": []
    }

    gammas = np.linspace(0, 1, num_gammas)
    results["gammas"] = gammas
    os.makedirs(os.path.join(directory, 'predictions'), exist_ok=True)
    for i, gamma in enumerate(gammas):
        # combine instance predictions according to gamma
        # gamma closer to 1: s-tree impacts prediction more
        # gamma closer to 0: y-tree impacts prediction more

        if predict_type == "predict_proba":
            preds_train = combined_predict_proba(y_preds_train, fair_preds_train, gamma)
            preds_test = combined_predict_proba(y_preds_test, fair_preds_test, gamma)
        else:
            preds_train = combined_predict(y_preds_train, fair_preds_train, gamma)
            preds_test = combined_predict(y_preds_test, fair_preds_test, gamma)

        pd.DataFrame({'preds_train': preds_train}).to_csv(
            os.path.join(directory, 'predictions', 'preds_train_fold{}_gamma{}.csv'.format(k_cv, gamma)))
        pd.DataFrame({'preds_test': preds_test}).to_csv(
            os.path.join(directory, 'predictions', 'preds_test_fold{}_gamma{}.csv'.format(k_cv, gamma)))

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

    return results


if __name__ == "__main__":
    main()

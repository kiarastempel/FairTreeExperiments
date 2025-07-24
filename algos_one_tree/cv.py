import numpy as np
import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split, KFold
from algos_two_trees.get_data import *
from sklearn.metrics import roc_auc_score, accuracy_score
from algos_two_trees.utils import statistical_parity_diff
from constants import PROJECT_ROOT
from algos_one_tree.method import create_tree
import warnings
import time


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

    parser.add_argument('--tree_variant', type=str, default='threshold_gain_s',
                        help='Type of tree that will be used as "fair" tree, has to be in '
                             '["threshold_gain_s", "weighted_combi", "backtracking"]')

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

    current_time = time.strftime("%H:%M:%S")
    print("Current time:", current_time)

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

    directory = os.path.join(PROJECT_ROOT, 'results_cv', str(args.data), 'tradeoffs_one_tree',
                             str(args.tree_variant), "seed_" + str(args.seed),
                             "min_samples_" + str(args.min_samples), "max_depth_" + str(args.max_depth))
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

        create_trees(X_train_cv, X_val_cv, y_train_cv, y_val_cv, s_train_cv, s_val_cv, unprivileged_group, pos_outcome,
                     args.predict_type, args.data, args.tree_variant, args.leaf_outcome_method, args.max_depth,
                     min_samples, args.num_gammas, k, directory)

        k += 1


def create_trees(X_train, X_test, y_train, y_test, s_train, s_test, unprivileged_group, pos_outcome, predict_type,
                 data, tree_variant, leaf_outcome_method, max_depth, min_samples, num_gammas, k_cv,
                 directory):
    # gamma sweep: create tree for each gamma and evaluate tree
    results = {
        "gammas": [],
        "spds_train": [],
        "spds_test": [],
        "aurocs_train": [],
        "aurocs_test": [],
        "accs_train": [],
        "accs_test": []
    }
    if tree_variant == "threshold_gain_s" or tree_variant == "backtracking":
        gammas = np.linspace(0, 0.2, num_gammas)  # i.e., use threshold
    elif tree_variant == "weighted_combi":
        gammas = np.linspace(0, 1, num_gammas)
    else:
        raise ValueError("given variant for training the tree is unknown")

    results["gammas"] = gammas
    os.makedirs(os.path.join(directory, 'predictions'), exist_ok=True)
    for gamma in gammas:
        # create tree and get its predictions
        preds_train, preds_test = create_tree(X_train, y_train, s_train, X_test, y_test, tree_variant, predict_type,
                                              max_depth, min_samples, leaf_outcome_method, unprivileged_group, pos_outcome,
                                              gamma, print_trees=False)

        if preds_train is not None:
            pd.DataFrame({'preds_train': preds_train}).to_csv(
                os.path.join(directory, 'predictions', 'preds_train_fold{}_gamma{}.csv'.format(k_cv, gamma)))
        else:
            pd.DataFrame({'preds_train': [preds_train]}).to_csv(
                os.path.join(directory, 'predictions', 'preds_train_fold{}_gamma{}.csv'.format(k_cv, gamma)))

        if preds_test is not None:
            pd.DataFrame({'preds_test': preds_test}).to_csv(
                os.path.join(directory, 'predictions', 'preds_test_fold{}_gamma{}.csv'.format(k_cv, gamma)))
        else:
            pd.DataFrame({'preds_test': [preds_test]}).to_csv(
                os.path.join(directory, 'predictions', 'preds_test_fold{}_gamma{}.csv'.format(k_cv, gamma)))

        if preds_train is None:
            # empty tree
            results["spds_train"].append(-1)
            results["spds_test"].append(-1)
            results["aurocs_train"].append(-1)
            results["aurocs_test"].append(-1)
            results["accs_train"].append(-1)
            results["accs_test"].append(-1)
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
    pd.DataFrame(results).to_csv(
        os.path.join(directory, 'results_fold{}.csv'.format(k_cv)))


if __name__ == "__main__":
    main()

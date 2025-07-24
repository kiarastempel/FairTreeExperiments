import numpy as np
import pandas as pd
import os
import argparse
import copy
from sklearn.model_selection import train_test_split
from algos_two_trees.get_data import *
from sklearn.metrics import roc_auc_score, accuracy_score
from FairTree.fair_classification_tree import FairClassificationTree
from algos_two_trees.utils import statistical_parity_diff
from constants import PROJECT_ROOT
import time


seed = 5


def main():
    parser = argparse.ArgumentParser(description='Get run values.')

    parser.add_argument('--data', type=str, default='Adult',
                        help='Dataset to use: Compas, Adult, Banks, German, Law, Dutch')

    parser.add_argument('--predict_type', type=str, default='predict_proba',
                        help='Type of prediction, has to be in ["predict", "predict_proba"]')

    parser.add_argument('--tree_variant', type=str, default='backtracking',
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

    parser.add_argument('--timed-run', action='store_true',
                        help='Do a timed run, ignoring most disk I/O')
    
    parser.add_argument('--timed-ncols', default=20, type=int)

    parser.add_argument('--timed-nrows', default=100, type=int)

    parser.add_argument('--print_trees', action='store_true',
                        help='Print the trees for interpretation')

    args = parser.parse_args()

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

    os.makedirs(os.path.join(PROJECT_ROOT, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, 'results', 'tradeoffs_one_tree'), exist_ok=True)

    X = X.applymap(lambda x: int(x) if isinstance(x, bool) else x).astype(np.float32)

    if args.timed_run:
        X, y, s = X.iloc[:args.timed_nrows, :args.timed_ncols], y[:args.timed_nrows], s[:args.timed_nrows]
    
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, pd.DataFrame(s), test_size=0.3, random_state=seed
    )

    min_samples = int(len(X_train) * args.min_samples / 100)

    create_trees(X_train, X_test, y_train, y_test, s_train, s_test, unprivileged_group, pos_outcome, args.predict_type,
                 args.data, args.tree_variant, args.leaf_outcome_method, args.max_depth, min_samples,
                 args.num_gammas, args.timed_run, args.timed_ncols, args.timed_nrows, args.print_trees)


def create_trees(X_train, X_test, y_train, y_test, s_train, s_test, unprivileged_group, pos_outcome, predict_type,
                 data, tree_variant, leaf_outcome_method, max_depth, min_samples, num_gammas, timed_run, n_cols, n_rows,
                 print_trees):
    start = time.time()
    if timed_run:
        start = time.time()
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
        gammas = [1.0]  # np.linspace(0, 0.5, num_gammas)  # i.e., use threshold
    elif tree_variant == "weighted_combi":
        gammas = np.linspace(0, 1, num_gammas)
    else:
        raise ValueError("given variant for training the tree is unknown")
    
    results["gammas"] = gammas
    for gamma in gammas:
        print(gamma)
        # create tree and get its predictions
        preds_train, preds_test = create_tree(X_train, y_train, s_train, X_test, y_test, tree_variant, predict_type,
                                              max_depth, min_samples, leaf_outcome_method, unprivileged_group, pos_outcome,
                                              gamma, print_trees)

        if preds_train is None:
            # emptry tree
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
    if timed_run:
        end = time.time()
        file_path = os.path.join(PROJECT_ROOT, 'results', 'tradeoffs_one_tree', 'time_{}_{}_{}_5.csv'.format(data, predict_type, tree_variant))
        header = not os.path.exists(file_path)
        time_results = pd.DataFrame({'time elapsed': [end - start], 'number of columns': n_cols,
                                     'number of rows': n_rows, 'max depth': max_depth})
        time_results.to_csv(file_path, mode='a', header=header, index=False)
    else:
        pd.DataFrame(results).to_csv(
            os.path.join(PROJECT_ROOT, 'results', 'tradeoffs_one_tree', 'results_{}_{}_{}_5.csv'.format(
                data, predict_type, tree_variant)))


def create_tree(X_train, y_train, s_train, X_test, y_test, tree_variant, predict_type, max_depth, min_samples,
                leaf_outcome_method, unprivileged_group, pos_outcome, sens_threshold, print_trees):
    X_train_fair = copy.deepcopy(X_train)
    X_train_fair['y'] = y_train
    X_test_fair = copy.deepcopy(X_test)
    X_test_fair['y'] = y_test
    cols = X_train.columns.tolist()

    if predict_type == "predict_proba":
        leaf_outcome_method = "probability"
    if tree_variant == "threshold_gain_s":
        fair_tree = FairClassificationTree(data=X_train_fair, attributes=cols, idx_target=-1,
                                           unprivileged_group=unprivileged_group, pos_outcome=pos_outcome,
                                           threshold_binning=10,
                                           sensitive=np.array(s_train.values).flatten().tolist(),
                                           leaf_outcome=leaf_outcome_method, split_criterion="threshold_constraint",
                                           sens_threshold=sens_threshold)
        fair_tree.fit(max_depth=max_depth, min_samples_leave=min_samples)
        if print_trees:
            fair_tree.print_tree()
            print("\n")

        preds_train = np.array(fair_tree.predict(X_train_fair[cols]))
        preds_test = np.array(fair_tree.predict(X_test_fair[cols]))
    elif tree_variant == "weighted_combi":
        fair_tree = FairClassificationTree(data=X_train_fair, attributes=cols, idx_target=-1,
                                           unprivileged_group=unprivileged_group, pos_outcome=pos_outcome,
                                           threshold_binning=10,
                                           sensitive=np.array(s_train.values).flatten().tolist(),
                                           leaf_outcome=leaf_outcome_method, split_criterion="weighted_combi",
                                           gamma=sens_threshold)
        fair_tree.fit(max_depth=max_depth, min_samples_leave=min_samples)
        if print_trees:
            print("gamma", sens_threshold)
            fair_tree.print_tree()
            print("\n")
        preds_train = np.array(fair_tree.predict(X_train_fair[cols]))
        preds_test = np.array(fair_tree.predict(X_test_fair[cols]))
    elif tree_variant == "backtracking":
        fair_tree = FairClassificationTree(data=X_train_fair, attributes=cols, idx_target=-1,
                                           unprivileged_group=unprivileged_group, pos_outcome=pos_outcome,
                                           threshold_binning=10,
                                           sensitive=np.array(s_train.values).flatten().tolist(),
                                           leaf_outcome=leaf_outcome_method, split_criterion="information_gain",
                                           sens_threshold=sens_threshold, backtracking=True)
        fair_tree.fit(max_depth=max_depth, min_samples_leave=min_samples)
        if print_trees:
            print("gamma", sens_threshold)
            fair_tree.print_tree()
            print("\n")
        if fair_tree.tree is None:
            return None, None
        preds_train = np.array(fair_tree.predict(X_train_fair[cols]))
        preds_test = np.array(fair_tree.predict(X_test_fair[cols]))
    else:
        raise ValueError("given variant for training the fair tree is unknown")
    return np.argmax(preds_train >= 0.5, axis=1), np.argmax(preds_test >= 0.5, axis=1)


if __name__ == "__main__":
    main()

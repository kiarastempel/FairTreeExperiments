import numpy as np
import argparse
from algos_two_trees.get_data import *
from algos_two_trees.method import *
from constants import PROJECT_ROOT
import sys
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from evaluation.evaluation_metrics import autoc
from algos_one_tree.method import *


seed = 42


def main():

    num_cv = 3
    num_seeds = 15

    parser = argparse.ArgumentParser(description='Get run values.')

    parser.add_argument('--path_input', type=str, default='results_cv/Compas/tradeoffs_one_tree/threshold_gain_s',
                        help='Path to specific input results folder')

    parser.add_argument('--path_output', type=str,
                        default='results_best/Compas/tradeoffs_one_tree/threshold_gain_s',
                        help='Path to specific folder where the output will be saved')

    parser.add_argument('--data', type=str, default='Compas',
                        help='Dataset to use: Compas, Adult, Banks, German, Law, Dutch')

    parser.add_argument('--predict_type', type=str, default='predict_proba',
                        help='Type of prediction, has to be in ["predict", "predict_proba"]')

    parser.add_argument('--tree_variant', type=str, default='threshold_gain_s',
                        help='Type of tree that will be used as "fair" tree, has to be in '
                             '["threshold_gain_s", "weighted_combi", "backtracking"]')

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
                    autoc_scores.append(autoc(results['spds_test'], aurocs_test))
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
        print(f"Best hyperparameters: ", best_hps)
        print("autoc:", best_autoc, "\n")

        # retrain model using best hps on X_train got by holdout_seed, test on corresponding X_test
        results = {
            "gammas": [],
            "spds_train": [],
            "spds_test": [],
            "aurocs_train": [],
            "aurocs_test": [],
            "accs_train": [],
            "accs_test": []
        }
        if args.tree_variant == "threshold_gain_s" or args.tree_variant == "backtracking":
            gammas = np.linspace(0, 0.2, args.num_gammas)  # i.e., use threshold
        elif args.tree_variant == "weighted_combi":
            gammas = np.linspace(0, 1, args.num_gammas)
        else:
            raise ValueError("given variant for training the tree is unknown")

        best_min_samples = int(len(X_train) * best_hps["min_samples"] / 100)
        best_max_depth = best_hps["max_depth"]

        results["gammas"] = gammas
        for gamma in gammas:
            print(gamma)
            # create tree and get its predictions
            preds_train, preds_test = create_tree(X_train, y_train, s_train, X_test, y_test, args.tree_variant,
                                                  args.predict_type,
                                                  best_max_depth, best_min_samples, args.leaf_outcome_method, unprivileged_group,
                                                  pos_outcome,
                                                  gamma, print_trees=False)
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

        pd.DataFrame(results).to_csv(
            os.path.join(directory, 'best_results_seed_{}_min_samples_{}_max_depth_{}).csv'.format(
                holdout_seed, best_hps["min_samples"], best_hps["max_depth"])))


if __name__ == "__main__":
    main()

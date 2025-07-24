import numpy as np
import argparse
from algos_two_trees.get_data import *
from algos_two_trees.method import *
from constants import PROJECT_ROOT
import sys
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from evaluation.evaluation_metrics import autoc
from algos_one_tree.method import *
from fairlearn_relaxed_threshold_optimizer.postprocessing import ThresholdOptimizer


seed = 42


def main():

    num_cv = 3
    num_seeds = 15

    parser = argparse.ArgumentParser(description='Get run values.')

    parser.add_argument('--path_input', type=str, default='results_cv/Compas/relaxed_threshold_optimizer',
                        help='Path to specific input results folder')

    parser.add_argument('--path_output', type=str,
                        default='results_best/Compas/relaxed_threshold_optimizer',
                        help='Path to specific folder where the output will be saved')

    parser.add_argument('--data', type=str, default='Compas',
                        help='Dataset to use: Compas, Adult, Banks, German, Law, Dutch')

    parser.add_argument('--predict_type', type=str, default='predict_proba',
                        help='Type of prediction, has to be in ["predict", "predict_proba"]')

    parser.add_argument('--num_gammas', type=int, default='50',
                        help='Number of gamma values between 0 and 1')

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

    path = os.path.join(PROJECT_ROOT, args.path_input)
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
                    # calculate AUTOC for fold and save with hps in dict
                    autoc_scores.append(autoc(results['spds_test'], results['aurocs_test']))
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

        best_min_samples = int(len(X_train) * best_hps["min_samples"] / 100)
        best_max_depth = best_hps["max_depth"]

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

        tolerances = np.linspace(0, 0.25, args.num_gammas + 1)[1:]
        results["gammas"] = tolerances
        for tol in tolerances:
            print(tol)

            # retrain threshold optimizer
            unmitigated_tree = DecisionTreeClassifier(max_depth=best_max_depth, min_samples_leaf=best_min_samples) \
                .fit(X_train, y_train)

            postprocesser = ThresholdOptimizer(
                estimator=unmitigated_tree,
                constraints="demographic_parity",
                objective="accuracy_score",
                grid_size=1000,
                prefit=True,
                predict_method='predict_proba',
                tol=tol,
                tol_method='to_overall'
            )

            postprocesser.fit(X_train, y_train, sensitive_features=s_train)

            # Make predictions on the test set
            preds_train = postprocesser.predict(X_train, sensitive_features=s_train)
            preds_test = postprocesser.predict(X_test, sensitive_features=s_test)

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

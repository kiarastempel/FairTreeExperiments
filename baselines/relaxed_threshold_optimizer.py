import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from algos_two_trees.get_data import *
from constants import PROJECT_ROOT
from algos_two_trees.utils import statistical_parity_diff
from sklearn.metrics import roc_auc_score, accuracy_score
from fairlearn_relaxed_threshold_optimizer.postprocessing import ThresholdOptimizer


seed = 42


def main():
    parser = argparse.ArgumentParser(description='Get run values.')

    parser.add_argument('--data', type=str, default='Compas',
                        help='Dataset to use: Compas, Adult, Banks, German')

    args = parser.parse_args()

    if args.data == "Compas":
        X, y, s, unprivileged_group, pos_outcome = get_compas(
            os.path.join(PROJECT_ROOT, 'data', 'compas-preprocessed.csv'))
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

    os.makedirs(os.path.join(PROJECT_ROOT, 'results', 'relaxed_threshold_optimizer'), exist_ok=True)

    X = X.applymap(lambda x: int(x) if isinstance(x, bool) else x).astype(np.float32)

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, pd.DataFrame(s), test_size=0.3, random_state=seed
    )

    results = {
        "gammas": [],
        "spds_train": [],
        "spds_test": [],
        "aurocs_train": [],
        "aurocs_test": [],
        "accs_train": [],
        "accs_test": []
    }

    tolerances = np.linspace(0, 0.25, 51)[1:]
    results["gammas"] = tolerances

    for tol in tolerances:
        print(tol)
        # Train threshold optimizer for postprocessing
        unmitigated_tree = DecisionTreeClassifier().fit(X_train, y_train)

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

    # save results to csv
    pd.DataFrame(results).to_csv(
        os.path.join(PROJECT_ROOT, 'results', 'relaxed_threshold_optimizer', 'relaxed_threshold_optimizer_{}.csv'.format(args.data)))


if __name__ == "__main__":
    main()

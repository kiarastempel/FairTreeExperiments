import argparse
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from algos_two_trees.utils import statistical_parity_diff
from sklearn.metrics import roc_auc_score, accuracy_score
from algos_two_trees.get_data import *
from constants import PROJECT_ROOT
import copy
from fair_classification_tree import FairClassificationTree


seed = 42


def main():
    parser = argparse.ArgumentParser(description='Get run values.')

    parser.add_argument('--data', type=str, default='Folktables_HI',
                        help='Dataset to use: Compas, Adult, Banks, German, Dutch, Law')

    args = parser.parse_args()

    X, y, s, unprivileged_group, pos_outcome = data_loader_router(args.data, intersectional=False)

    os.makedirs(os.path.join(PROJECT_ROOT, 'results_baselines', 'randomized_classifier'), exist_ok=True)

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

    # for prediction: with a probability of (1-p)%, use the model prediction, with p%, just decide randomly
    probs = np.linspace(0, 1, 50)
    results["gammas"] = probs

    # Calculate the step size for adding random predictions
    step_size = len(X_test) // len(probs)

    for p in probs:
        print("p", p)

        X_train_fair = copy.deepcopy(X_train)
        X_train_fair['y'] = y_train
        X_test_fair = copy.deepcopy(X_test)
        X_test_fair['y'] = y_test
        cols = X_train.columns.tolist()

        y_tree = FairClassificationTree(data=X_train_fair, attributes=cols, idx_target=-1,
                                        unprivileged_group=unprivileged_group, pos_outcome=pos_outcome,
                                        threshold_binning=10,
                                        sensitive=np.array(s_train.values).flatten().tolist(),
                                        leaf_outcome="majority", split_criterion="information_gain")
        y_tree.fit(max_depth=4, min_samples_leave=5, tree_type="performance")

        preds_train = np.array(y_tree.predict(X_train_fair[cols]))
        preds_test = np.array(y_tree.predict(X_test_fair[cols]))

        # clf = DecisionTreeClassifier(max_depth=1)
        # clf.fit(X_train, y_train)
        # preds_train = clf.predict(X_train)
        # preds_test = clf.predict(X_test)

        # with probability p, change a prediction to a random prediction, i.e. randomly choose between 0 and 1
        for i in range(len(preds_test)):
            if np.random.rand() < p:
                # decide randomly
                prediction = np.random.choice([0, 1])
                preds_test[i] = prediction  # 1 - preds_test[0]

        for i in range(len(preds_train)):
            if np.random.rand() < p:
                # decide randomly
                prediction = np.random.choice([0, 1])
                preds_train[i] = prediction  # 1 - preds_train[0]

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
        os.path.join(PROJECT_ROOT, 'results_baselines', 'randomized_classifier', 'randomized_classifier_{}.csv'.format(args.data)))


if __name__ == "__main__":
    main()

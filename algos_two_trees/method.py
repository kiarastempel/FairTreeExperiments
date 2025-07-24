import numpy as np
import random
import argparse
import copy
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from algos_two_trees.get_data import *
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
from FairTree.fair_classification_tree import FairClassificationTree
from algos_two_trees.utils import statistical_parity_diff
from pydl85 import DL85Classifier, DL85Predictor
from constants import PROJECT_ROOT
import time


seed = 42
k = 1


def main():
    parser = argparse.ArgumentParser(description='Get run values.')

    parser.add_argument('--data', type=str, default='Compas',
                        help='Dataset to use: Compas, Adult, Banks, German, Law, Dutch')

    parser.add_argument('--predict_type', type=str, default='predict_proba',
                        help='Type of prediction, has to be in ["predict", "predict_proba"]')

    parser.add_argument('--fair_tree_variant', type=str, default='fair_ensemble',
                        help='Type of tree that will be used as "fair" tree, has to be in '
                             '["fairness_gain", "fair_ensemble", "optimal_gain_s", "optimal_spd"]')

    parser.add_argument('--num_fair_ensemble', type=int, default='1',
                        help='If fair_ensemble is chosen as tree variant for the fair tree: number of ensemble members')

    parser.add_argument('--performance_tree_variant', type=str, default='own',
                        help='Type of tree that will be used as "fair" tree, has to be in '
                             '["sklearn", "own", "optimal"]')

    parser.add_argument('--split_criterion', type=str, default='gain_s',
                        help='Indicates which method is used in split criterion for the fair tree, has to be in '
                             '["information_gain", "gain_s", "spd", "threshold_constraint"]')

    parser.add_argument('--leaf_outcome_method', type=str, default='majority',
                        help='Which value or model is saved in the leaves to give a tree output, has to be in '
                             '["mean", "majority"]')

    parser.add_argument('--max_depth', type=int, default='6',
                        help='Maximum depth of optimal tree (for deeper trees runtime increases)')

    parser.add_argument('--min_samples', type=int, default='25',
                        help='Minimum samples per leave')

    parser.add_argument('--max_h_y', type=float, default='0.01',
                        help="Maximum entropy/uncertainty that should be there regarding y in a leaf of the performance"
                             "tree")

    parser.add_argument('--min_h_s', type=float, default='0.01',
                        help="Minimum entropy/uncertainty that should be there regarding s in a leaf of the fair"
                             "tree")

    parser.add_argument('--num_gammas', type=int, default='50',
                        help='Number of gamma values between 0 and 1')

    parser.add_argument('--timed-run', action='store_true',
                        help='Do a timed run: therefore, ignore disk I/O during experiments')
    
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

    os.makedirs(os.path.join(PROJECT_ROOT, 'plots', 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, 'results', 'tradeoffs_two_trees'), exist_ok=True)

    X = X.applymap(lambda x: int(x) if isinstance(x, bool) else x).astype(np.float32)

    if args.timed_run:
        X, y, s = X.iloc[:args.timed_nrows, :args.timed_ncols], y[:args.timed_nrows], s[:args.timed_nrows]

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, pd.DataFrame(s), test_size=0.33, random_state=seed
    )

    combine_trees(X_train, X_test, y_train, y_test, s_train, s_test, unprivileged_group, pos_outcome, args.predict_type,
                  args.data, args.fair_tree_variant, args.num_fair_ensemble, args.performance_tree_variant,
                  args.leaf_outcome_method, args.split_criterion,
                  args.max_depth, args.min_samples, args.max_h_y, args.min_h_s,
                  args.num_gammas, args.timed_run, args.timed_ncols, args.timed_nrows, print_trees=True)


def combine_trees(X_train, X_test, y_train, y_test, s_train, s_test, unprivileged_group, pos_outcome, predict_type,
                  data, fair_tree_variant, num_fair_ensemble, performance_tree_variant, leaf_outcome_method,
                  split_criterion, max_depth, min_samples, max_h_y, min_h_s,
                  num_gammas, timed_run, n_cols, n_rows, print_trees):
    X_train, y_train, s_train = resample(X_train, y_train, s_train, replace=True, n_samples=5000, random_state=seed)
    if timed_run:
        start_time = time.time()
    # get predictions of both separate trees
    y_preds_train, y_preds_test = create_performance_tree(X_train, y_train, s_train, X_test, y_test,
                                                          performance_tree_variant, predict_type,
                                                          max_depth, min_samples, max_h_y, leaf_outcome_method,
                                                          unprivileged_group, pos_outcome, split_criterion, print_trees)
    # todo: create fair ensemble instead using bootstrapping
    fair_preds_train, fair_preds_test = create_fair_tree(X_train, y_train, s_train, X_test, y_test, fair_tree_variant,
                                                         num_fair_ensemble, predict_type,
                                                         max_depth, min_samples, min_h_s, leaf_outcome_method,
                                                         unprivileged_group, pos_outcome, split_criterion, print_trees)

    # gamma sweep: combine separate predictions using various gamma values
    results = gamma_sweep(predict_type, y_preds_train, y_preds_test, fair_preds_train, fair_preds_test, s_train, s_test,
                          y_train, y_test, unprivileged_group, pos_outcome, data, fair_tree_variant,
                          performance_tree_variant, split_criterion, num_gammas, timed_run, print_trees)
    
    # save results to csv
    if timed_run:
        end_time = time.time()
        time_results = pd.DataFrame({'time elapsed': [end_time - start_time],
                                     'number of cols': n_cols, 'number of rows': n_rows,
                                     'max_depth': max_depth})
        file_path = os.path.join(PROJECT_ROOT, 'results', 'tradeoffs_two_trees', 'time_{}_{}_FAIR{}_PERF{}_({}).csv'.format(
            data, predict_type, fair_tree_variant, performance_tree_variant, split_criterion))
        header = not os.path.exists(file_path) 
        time_results.to_csv(file_path, mode='a', header=header, index=False)
    else:
        pd.DataFrame(results).to_csv(
            os.path.join(PROJECT_ROOT, 'results', 'tradeoffs_two_trees', 'h_limit_ensemble_results_k_{}_{}_{}_FAIR{}_PERF{}_({}).csv'.format(
                k, data, predict_type, fair_tree_variant, performance_tree_variant, split_criterion)))
        
 
def create_performance_tree(X_train, y_train, s_train, X_test, y_test, tree_variant, predict_type, max_depth,
                            min_samples, max_h_y,
                            leaf_outcome_method, unprivileged_group, pos_outcome, split_criterion, print_trees):
    if tree_variant == "sklearn":
        y_tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples)
        y_tree.fit(X_train, y_train)
        if predict_type == "predict_proba":
            y_preds_train = y_tree.predict_proba(X_train)
            y_preds_test = y_tree.predict_proba(X_test)
        else:
            y_preds_train = y_tree.predict(X_train)
            y_preds_test = y_tree.predict(X_test)
        if print_trees:
            tree_text = export_text(y_tree)
            print("performance tree")
            print(tree_text)
    elif tree_variant == "own":
        X_train_fair = copy.deepcopy(X_train)
        X_train_fair['y'] = y_train
        X_test_fair = copy.deepcopy(X_test)
        X_test_fair['y'] = y_test
        cols = X_train.columns.tolist()

        if predict_type == "predict_proba":
            leaf_outcome_method = "probability"

        y_tree = FairClassificationTree(data=X_train_fair, attributes=cols, idx_target=-1,
                                        unprivileged_group=unprivileged_group, pos_outcome=pos_outcome,
                                        threshold_binning=10,
                                        sensitive=np.array(s_train.values).flatten().tolist(),
                                        leaf_outcome=leaf_outcome_method, split_criterion="information_gain")
        y_tree.fit(max_depth=max_depth, min_samples_leave=min_samples, type="performance", h_limit=max_h_y)
        if print_trees:
            print("performance tree")
            y_tree.print_tree()

        y_preds_train = np.array(y_tree.predict(X_train_fair[cols]))
        y_preds_test = np.array(y_tree.predict(X_test_fair[cols]))

    elif tree_variant == "optimal":
        clf = DL85Classifier(max_depth=max_depth, min_sup=min_samples)
        clf.fit(X_train, y_train)
        if predict_type == "predict_proba":
            y_preds_train = clf.predict_proba(X_train)
            y_preds_test = clf.predict_proba(X_test)
        else:
            y_preds_train = clf.predict(X_train)
            y_preds_test = clf.predict(X_test)
    else:
        raise ValueError("given variant (" + str(tree_variant) + ") for training the performance tree is unknown")

    return y_preds_train, y_preds_test


def create_fair_tree(X_train, y_train, s_train, X_test, y_test, tree_variant, num_fair_ensemble, predict_type,
                     max_depth, min_samples, min_h_s,
                     leaf_outcome_method, unprivileged_group, pos_outcome, split_criterion, print_trees):
    if tree_variant == "fairness_gain":
        X_train_fair = copy.deepcopy(X_train)
        X_train_fair['y'] = y_train
        X_test_fair = copy.deepcopy(X_test)
        X_test_fair['y'] = y_test
        cols = X_train.columns.tolist()

        if predict_type == "predict_proba":
            leaf_outcome_method = "probability"

        fair_tree = FairClassificationTree(data=X_train_fair, attributes=cols, idx_target=-1,
                                           unprivileged_group=unprivileged_group, pos_outcome=pos_outcome,
                                           threshold_binning=10,
                                           sensitive=np.array(s_train.values).flatten().tolist(),
                                           leaf_outcome=leaf_outcome_method, split_criterion=split_criterion)
        fair_tree.fit(max_depth=max_depth, min_samples_leave=min_samples, type="fair", h_limit=min_h_s)
        if print_trees:
            print("fair tree")
            fair_tree.print_tree()

        fair_preds_train = np.array(fair_tree.predict(X_train_fair[cols]))
        fair_preds_test = np.array(fair_tree.predict(X_test_fair[cols]))
    elif tree_variant == "fair_ensemble":
        print("creating ensemble")
        X_train_fair = copy.deepcopy(X_train)
        X_train_fair['y'] = y_train
        X_test_fair = copy.deepcopy(X_test)
        X_test_fair['y'] = y_test
        cols = X_train.columns.tolist()
        # bootstrap
        # k (=num_fair_ensemble) times, sample n many instances from the dataset
        ensemble = []
        predictions_train = []
        predictions_test = []
        for b in range(num_fair_ensemble):
            print("tree ", b)
            X_boot, y_boot, s_boot = resample(X_train, y_train, s_train, replace=True, random_state=b)
            # transform these into needed dataframe format
            X_train_boot = copy.deepcopy(X_boot)
            X_train_boot['y'] = y_boot
            cols = X_boot.columns.tolist()
            # call fair_tree
            fair_tree = FairClassificationTree(data=X_train_boot, attributes=cols, idx_target=-1,
                                               unprivileged_group=unprivileged_group, pos_outcome=pos_outcome,
                                               threshold_binning=10,
                                               sensitive=np.array(s_boot.values).flatten().tolist(),
                                               leaf_outcome="probability", split_criterion=split_criterion)
            fair_tree.fit(max_depth=max_depth, min_samples_leave=min_samples, type="fair")
            # save tree to list of trees (=ensemble)
            ensemble.append(fair_tree)
            predictions_train.append(np.array(fair_tree.predict(X_train_boot[cols])))
            predictions_test.append(np.array(fair_tree.predict(X_test_fair[cols])))

        # result: ensemble of k fair trees

        # predict: average the class probabilities
        fair_preds_train = np.mean(predictions_train, axis=0)
        fair_preds_test = np.mean(predictions_test, axis=0)

    elif tree_variant == "optimal_gain_s":
        fair_tree = get_optimal_tree_gain_s(X_train, y_train, s_train, max_depth)
        if predict_type == "predict":
            fair_preds_train = fair_tree.predict(X_train)
            fair_preds_test = fair_tree.predict(X_test)
        else:
            # todo: doesn't work?
            fair_preds_train = fair_tree.predict_proba(X_train)
            fair_preds_test = fair_tree.predict_proba(X_test)
    elif tree_variant == "optimal_spd":
        fair_tree = get_optimal_tree_spd(X_train, y_train, s_train, max_depth, unprivileged_group, pos_outcome)
        if predict_type == "predict":
            fair_preds_train = fair_tree.predict(X_train)
            fair_preds_test = fair_tree.predict(X_test)
        else:
            # todo: doesn't work?
            fair_preds_train = fair_tree.predict_proba(X_train)
            fair_preds_test = fair_tree.predict_proba(X_test)
    else:
        raise ValueError("given variant (" + str(tree_variant) + ") for training the fair tree is unknown")
    return fair_preds_train, fair_preds_test


def get_optimal_tree_gain_s(X_train, y_train, s_train, max_depth):
    def error_gain_s(tids):
        # minimize accuracy on s, i.e. maximize error on s
        classes, supports = np.unique(s_train.take(list(tids)), return_counts=True)
        maxindex = np.argmax(supports)
        return -1 * (sum(supports) - supports[maxindex])

    def leaf_value(tids):
        classes, supports = np.unique(y_train.take(list(tids)), return_counts=True)
        maxindex = np.argmax(supports)
        return classes[maxindex]

    fair_tree = DL85Predictor(error_function=error_gain_s,
                              leaf_value_function=leaf_value, max_depth=max_depth)  # , min_sup=5,
    fair_tree.fit(X_train, y_train)
    return fair_tree


def get_optimal_tree_spd(X_train, y_train, s_train, max_depth, unprivileged_group, pos_outcome):
    def error_spd(tids):
        # minimize statistical parity difference
        classes, supports = np.unique(y_train.take(list(tids)), return_counts=True)
        maxindex = np.argmax(supports)
        y_preds = [int(classes[maxindex])] * len(y_train)
        spd = statistical_parity_diff(y_preds, s_train.values.flatten(), unprivileged_group, pos_outcome)
        return spd

    def leaf_value(tids):
        classes, supports = np.unique(y_train.take(list(tids)), return_counts=True)
        maxindex = np.argmax(supports)
        return classes[maxindex]

    fair_tree = DL85Predictor(error_function=error_spd,
                              leaf_value_function=leaf_value, max_depth=max_depth)  # , min_sup=5,
    fair_tree.fit(X_train, y_train)
    return fair_tree


def combined_predict(y_preds, s_preds, gamma):
    preds = (1 - gamma) * y_preds + gamma * np.array(s_preds)
    return preds >= 0.5


def combined_predict_proba(y_preds, s_preds, gamma):
    preds_1 = (1 - gamma) * y_preds
    preds_2 = gamma * s_preds
    preds = (1 - gamma) * y_preds + gamma * s_preds
    return np.argmax(preds >= 0.5, axis=1)


def gamma_sweep(predict_type, y_preds_train, y_preds_test, fair_preds_train, fair_preds_test, s_train, s_test,
                y_train, y_test, unprivileged_group, pos_outcome, data, fair_tree_variant, performance_tree_variant,
                split_criterion, num_gammas, timed_run, print_trees):
    results = {
        "gammas": [],
        "spds_train": [],
        "spds_test": [],
        "aurocs_train": [],
        "aurocs_test": [],
        "accs_train": [],
        "accs_test": [],
        "num_pred_changes_train": [0],
        "avg_perf_prob_train": [0],
        "avg_fair_prob_train": [0],
        "num_pred_changes_test": [0],
        "avg_perf_prob_test": [0],
        "avg_fair_prob_test": [0]
    }
    previous_preds_train = []
    previous_preds_test = []

    if not timed_run:
        fig, ax = plt.subplots(num_gammas + 2, 2, figsize=(15, 200))
        plt.subplots_adjust(hspace=0.5)

    if not timed_run:
        # this is only for looking at the distributions of the predictions, for a better understanding of the curve
        if predict_type == "predict_proba":
            ax[0, 0].hist(y_preds_train, label=["class 0", "class 1"])
            ax[0, 0].legend()
            ax[0, 0].set_title("Probabilities of fairness-unaware tree")

            ax[0, 1].hist(fair_preds_train, label=["class 0", "class 1"])
            ax[0, 1].legend()
            ax[0, 1].set_title("Probabilities of fairness-aware tree")

            ax[1, 0].hist(np.argmax(np.array(y_preds_train) >= 0.5, axis=1))
            ax[1, 0].legend()
            ax[1, 0].set_title("Predictions of fairness-unaware tree")

            ax[1, 1].hist(np.argmax(np.array(fair_preds_train) >= 0.5, axis=1))
            ax[1, 1].legend()
            ax[1, 1].set_title("Predictions of fairness-aware tree")

    gammas = np.linspace(0, 1, num_gammas)
    results["gammas"] = gammas
    for i, gamma in enumerate(gammas):
        if print_trees:
            print("iteration", str(i))
            print("gamma", gamma)
        # combine instance predictions according to gamma
        # gamma closer to 1: s-tree impacts prediction more
        # gamma closer to 0: y-tree impacts prediction more

        if predict_type == "predict_proba":
            preds_train = combined_predict_proba(y_preds_train, fair_preds_train, gamma)
            preds_test = combined_predict_proba(y_preds_test, fair_preds_test, gamma)
        else:
            preds_train = combined_predict(y_preds_train, fair_preds_train, gamma)
            preds_test = combined_predict(y_preds_test, fair_preds_test, gamma)

        if not timed_run:
            if predict_type == "predict_proba":
                ax[i + 2, 0].hist(preds_train, label="fairness-aware")
                ax[i + 2, 0].legend()
                ax[i + 2, 0].set_title("Combined Predictions | gamma = " + str(gamma))

        if not timed_run:
            if i > 0:
                # compare previous and current predictions for train
                preds_changed_train = np.sum(preds_train != previous_preds_train)
                results["num_pred_changes_train"].append(preds_changed_train)

                changed_indizes_train = np.where(preds_train != previous_preds_train)
                y_preds_changed_train = y_preds_train[changed_indizes_train]
                fair_preds_changed_train = fair_preds_train[changed_indizes_train]
                if predict_type == "predict_proba":
                    results["avg_perf_prob_train"].append(np.mean(y_preds_changed_train[:, pos_outcome]))
                    results["avg_fair_prob_train"].append(np.mean(fair_preds_changed_train[:, pos_outcome]))
                else:
                    results["avg_perf_prob_train"].append(y_preds_changed_train)
                    results["avg_fair_prob_train"].append(fair_preds_changed_train)

                # and for test
                preds_changed_test = np.sum(preds_test != previous_preds_test)
                results["num_pred_changes_test"].append(preds_changed_test)

                changed_indizes_test = np.where(preds_test != previous_preds_test)
                y_preds_changed_test = y_preds_test[changed_indizes_test]
                fair_preds_changed_test = fair_preds_test[changed_indizes_test]
                if predict_type == "predict_proba":
                    results["avg_perf_prob_test"].append(np.mean(y_preds_changed_test[:, pos_outcome]))
                    results["avg_fair_prob_test"].append(np.mean(fair_preds_changed_test[:, pos_outcome]))
                else:
                    results["avg_perf_prob_test"].append(y_preds_changed_test)
                    results["avg_fair_prob_test"].append(fair_preds_changed_test)

        # update previous predictions
        previous_preds_train = preds_train
        previous_preds_test = preds_test

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

    if not timed_run:
        plt.savefig(os.path.join(PROJECT_ROOT, 'plots', 'predictions',
                                'predictions_{}_{}_FAIR{}_PERF{}_({})_18.pdf'.format(data, predict_type, fair_tree_variant,
                                                                                performance_tree_variant,
                                                                                split_criterion)))

    return results


if __name__ == "__main__":
    main()


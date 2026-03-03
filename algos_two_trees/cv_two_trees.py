import shutil
import subprocess
import sys
from pathlib import Path

sys.path.append('../algos_two_trees')
sys.path.append('../algos_one_tree')
sys.path.append('../FairTree')
sys.path.append('../')

import numpy as np
import pandas as pd
import os
import argparse
import copy
from sklearn.model_selection import train_test_split, KFold
from get_data import *
from sklearn.metrics import roc_auc_score, accuracy_score
from utils import statistical_parity_diff
from fairlearn.metrics import demographic_parity_difference
from constants import PROJECT_ROOT
from method import create_performance_tree, create_fair_tree, combined_predict_proba, combined_predict
import warnings
import optuna
from FairTree.fair_classification_tree import FairClassificationTree


def main():
    parser = argparse.ArgumentParser(description='Get run values.')

    parser.add_argument('--data', type=str, default='Compas',
                        help='Dataset to use: Compas, Adult, Banks, German, Law, Dutch, Folktables_AK, Folktables_HI')

    parser.add_argument('--results_path', type=str, default='default',
                        help='"default" or custom path')

    parser.add_argument('--seed', type=int, default='42',
                        help='Seed for holdout')

    parser.add_argument('--num_cv', type=int, default='3',
                        help='Number of folds for cross-validation')

    parser.add_argument('--predict_type', type=str, default='predict_proba',
                        help='Type of prediction, has to be in ["predict", "predict_proba"]')

    parser.add_argument('--fair_tree_variant', type=str, default='fairness_gain',
                        help='Type of tree that will be used as "fair" tree, has to be in '
                             '["fairness_gain", "optimal_gain_s", "optimal_spd"]')

    parser.add_argument('--num_fair_ensemble', type=int, default='1',
                        help='If fair_ensemble is chosen as tree variant for the fair tree: number of ensemble members')

    parser.add_argument('--performance_tree_variant', type=str, default='own',
                        help='Type of tree that will be used as "fair" tree, has to be in '
                             '["sklearn", "own", "performance_ensemble", "optimal", "backtracking"]')

    parser.add_argument('--num_performance_ensemble', type=int, default='1',
                        help='If performance_ensemble is chosen as tree variant for the performance tree: '
                             'number of ensemble members')

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

    parser.add_argument('--max_h_y', type=float, default='1.1',
                        help="Maximum entropy/uncertainty that should be there regarding y in a leaf of the performance"
                             "tree")

    parser.add_argument('--min_h_s', type=float, default='0.0',
                        help="Minimum entropy/uncertainty that should be there regarding s in a leaf of the fair"
                             "tree")

    parser.add_argument('--num_gammas', type=int, default='50',
                        help='Number of gamma values between 0 and 1')

    parser.add_argument('--intersectional', action='store_true',
                        help='Employ multiple sensitive attributes in data loaders')

    parser.add_argument('--combination', type=str, default='meta_tree',
                        help='How the predictions of the performance tree and the fair tree are combined '
                             '["linear", "meta_tree", "meta_tree_optimization"]')

    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    X, y, s, unprivileged_group, pos_outcome = data_loader_router(args.data, args.intersectional)

    if args.results_path == "default":
        directory = os.path.join(PROJECT_ROOT, 'results_cv', str(args.data),
                                 'tradeoffs_two_tree',
                                 str(args.combination) + "_AND_" + str(args.fair_tree_variant) + "_AND_" + str(args.performance_tree_variant),
                                 "seed_" + str(args.seed),
                                 "min_samples_" + str(args.min_samples),
                                 "max_depth_" + str(args.max_depth),
                                 "max_h_y_" + str(args.max_h_y),
                                 "min_h_s" + str(args.min_h_s))
    else:
        directory = os.path.join(args.results_path, 'results_cv', str(args.data),
                                 'tradeoffs_two_tree',
                                 str(args.combination) + "_AND_" + str(args.fair_tree_variant) + "_AND_" + str(args.performance_tree_variant),
                                 "seed_" + str(args.seed),
                                 "min_samples_" + str(args.min_samples),
                                 "max_depth_" + str(args.max_depth),
                                 "max_h_y_" + str(args.max_h_y),
                                 "min_h_s" + str(args.min_h_s))
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
                      args.data, args.fair_tree_variant, args.num_fair_ensemble, args.performance_tree_variant,
                      args.num_performance_ensemble, args.leaf_outcome_method, args.split_criterion,
                      args.max_depth, min_samples, args.max_h_y, args.min_h_s, args.num_gammas, k, directory,
                      args.intersectional, args.combination)

        k += 1

        # zip predictions folder and remove original one to have less saved files
        pred_dir = Path(directory) / "predictions"
        tar_path = pred_dir.with_suffix(".tar.gz")
        if pred_dir.exists():
            subprocess.run(
                ["tar", "-czf", tar_path, "-C", directory, "predictions"],
                check=True,
            )
            shutil.rmtree(pred_dir)


def combine_trees(X_train, X_test, y_train, y_test, s_train, s_test, unprivileged_group, pos_outcome, predict_type,
                  data, fair_tree_variant, num_fair_ensemble, performance_tree_variant, num_performance_ensemble,
                  leaf_outcome_method, split_criterion, max_depth,
                  min_samples, max_h_y, min_h_s, num_gammas, k_cv, directory, intersectional=False,
                  combination="linear"):
    # get predictions of both separate trees
    y_preds_train, y_preds_test = create_performance_tree(X_train, y_train, s_train, X_test, y_test,
                                                          performance_tree_variant, num_performance_ensemble,
                                                          predict_type, max_depth, min_samples, max_h_y,
                                                          leaf_outcome_method,
                                                          unprivileged_group, pos_outcome, split_criterion,
                                                          print_trees=False)
    fair_preds_train, fair_preds_test = create_fair_tree(X_train, y_train, s_train, X_test, y_test, fair_tree_variant,
                                                         num_fair_ensemble,
                                                         predict_type, max_depth, min_samples, min_h_s,
                                                         leaf_outcome_method,
                                                         unprivileged_group, pos_outcome, split_criterion,
                                                         print_trees=False)

    # gamma sweep: combine separate predictions using various gamma values
    results = gamma_sweep(predict_type, y_preds_train, y_preds_test, fair_preds_train, fair_preds_test, s_train, s_test,
                          y_train, y_test, unprivileged_group, pos_outcome, max_depth, min_samples, num_gammas, k_cv,
                          directory, intersectional=intersectional, combination=combination)

    # save results to csv
    pd.DataFrame(results).to_csv(os.path.join(directory, 'results_fold{}.csv'.format(k_cv)))


def gamma_sweep(predict_type, y_preds_train, y_preds_test, fair_preds_train, fair_preds_test, s_train, s_test,
                y_train, y_test, unprivileged_group, pos_outcome, max_depth, min_samples,
                num_gammas, k_cv, directory, intersectional=False, combination="linear"):
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

    if combination in ["meta_tree_optimization", "meta_tree"]:
        # ----- split data into train and validation -----
        meta_X_train = pd.DataFrame(
            {"y_preds": [p[0] for p in y_preds_train], "fair_preds": [p[0] for p in fair_preds_train], 'y': y_train})
        meta_X_test = pd.DataFrame(
            {"y_preds": [p[0] for p in y_preds_test], "fair_preds": [p[0] for p in fair_preds_test], 'y': y_test})

        meta_train, meta_val, s_train_split, s_val_split, y_train_split, y_val_split = train_test_split(
            meta_X_train,
            s_train,
            y_train,
            test_size=0.5,
            random_state=42,
            stratify=meta_X_train["y"]
        )

        # ----- calculate AUROC of y_preds_test and SPD of y_fair_test to save as AUROC_optimal and SPD_optimal -----
        auroc_optimal = roc_auc_score(y_train, np.argmax(y_preds_train >= 0.5, axis=1))
        spd_optimal = statistical_parity_diff(
            np.argmax(fair_preds_train >= 0.5, axis=1),
            s_train.squeeze().to_numpy(),
            unprivileged_group=unprivileged_group,
            pos_outcome=pos_outcome
        )

        cols = ["y_preds", "fair_preds"]

    for i, gamma in enumerate(gammas):
        # combine instance predictions according to gamma
        # gamma closer to 1: s-tree impacts prediction more
        # gamma closer to 0: y-tree impacts prediction more

        if combination == "meta_tree_optimization":
            def objective(trial):
                # Suggest values for the hyperparameters
                max_depth_hp = trial.suggest_int("max_depth", 1, 10)
                min_samples_hp = trial.suggest_int("min_samples", 2, len(y_train_split) / 4)

                # ----- train tree -----
                model = FairClassificationTree(data=meta_val, attributes=cols, idx_target=-1,
                                               unprivileged_group=unprivileged_group, pos_outcome=pos_outcome,
                                               threshold_binning=10,
                                               sensitive=np.array(s_val_split.values).flatten().tolist(),
                                               leaf_outcome="probability", split_criterion="chebyshev",
                                               gamma=gamma)
                model.fit(max_depth=max_depth_hp, min_samples_leave=min_samples_hp)

                # ----- predict y for validation data -----
                y_preds_val = np.array(model.predict(meta_val))
                y_preds_val = np.argmax(y_preds_val >= 0.5, axis=1)

                # ----- calculate AUROC & SPD for validation data -----
                spd_val = statistical_parity_diff(y_preds_val, np.asarray(s_val_split), unprivileged_group, pos_outcome)
                auroc_val = roc_auc_score(y_val_split, y_preds_val)

                # calculate chebyshev formula and return as objective
                chebyshev = max(
                    (1 - gamma) * abs(auroc_optimal - auroc_val),
                    gamma * abs(spd_optimal - spd_val)
                )

                return chebyshev

            # create optuna study & choose optimizer and objective direction
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
            study.optimize(objective, n_trials=30, n_jobs=1)

            # choose best model: retrain model on best found hyperparameters
            meta_tree = FairClassificationTree(data=meta_val, attributes=cols, idx_target=-1,
                                               unprivileged_group=unprivileged_group, pos_outcome=pos_outcome,
                                               threshold_binning=10,
                                               sensitive=np.array(s_val_split.values).flatten().tolist(),
                                               leaf_outcome="probability", split_criterion="chebyshev",
                                               gamma=gamma)
            meta_tree.fit(max_depth=study.best_trial.params["max_depth"],
                          min_samples_leave=study.best_trial.params["min_samples"])
        elif combination == "meta_tree":
            meta_tree = FairClassificationTree(data=meta_val, attributes=cols, idx_target=-1,
                                               unprivileged_group=unprivileged_group, pos_outcome=pos_outcome,
                                               threshold_binning=10,
                                               sensitive=np.array(s_val_split.values).flatten().tolist(),
                                               leaf_outcome="probability", split_criterion="chebyshev",
                                               gamma=gamma)
            meta_tree.fit(max_depth=max_depth, min_samples_leave=min_samples)

        # if meta_tree, call combined_predict_proba using meta_tree
        if predict_type == "predict_proba" and combination in ["meta_tree_optimization", "meta_tree"]:
            preds_train = combined_predict_proba(y_preds_train, fair_preds_train, gamma, meta_tree=meta_tree)
            preds_test = combined_predict_proba(y_preds_test, fair_preds_test, gamma, meta_tree=meta_tree)
        elif predict_type == "predict_proba" and combination == "linear":
            preds_train = combined_predict_proba(y_preds_train, fair_preds_train, gamma)
            preds_test = combined_predict_proba(y_preds_test, fair_preds_test, gamma)
        elif predict_type == "predict" and combination == "linear":
            preds_train = combined_predict(y_preds_train, fair_preds_train, gamma)
            preds_test = combined_predict(y_preds_test, fair_preds_test, gamma)
        else:
            ValueError(f"Combination/predict variant ({combination}/{predict_type}) not implemented")

        pd.DataFrame({'preds_train': preds_train}).to_csv(
            os.path.join(directory, 'predictions', 'preds_train_fold{}_gamma{}.csv'.format(k_cv, gamma)))
        pd.DataFrame({'preds_test': preds_test}).to_csv(
            os.path.join(directory, 'predictions', 'preds_test_fold{}_gamma{}.csv'.format(k_cv, gamma)))

        if intersectional:
            spd_train = demographic_parity_difference(y_train, preds_train, sensitive_features=s_train)
            spd_test = demographic_parity_difference(y_test, preds_test, sensitive_features=s_test)
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

    return results


if __name__ == "__main__":
    main()

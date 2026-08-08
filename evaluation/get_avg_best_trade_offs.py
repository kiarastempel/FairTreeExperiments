import os
import sys

from constants import PROJECT_ROOT

sys.path.append('../algos_two_trees')
sys.path.append('../evaluation')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
import argparse
from get_data import *
import glob
import numpy as np
import pandas as pd
from evaluation_metrics import autoc, num_unique_points, \
    check_point_distribution, num_local_pareto_points, hypervolume, general_distance, deb_spread

seed = 42


def main():
    parser = argparse.ArgumentParser(description='Get run values.')

    parser.add_argument('--data', type=str, default='Dutch',
                        help='Dataset to use: Compas, Adult, Banks, German, Law, Dutch')

    parser.add_argument('--path_input', type=str, default='results_cv_SMS/results_best_pymoo_SMS2026-07-26',
                        help='Path to specific input results folder')

    parser.add_argument('--path_output', type=str, default='results_cv_SMS/results_best_pymoo_SMS2026-07-26',
                        help='Path to specific folder where the output will be saved')

    parser.add_argument('--fairness_metric', type=str, default='accuracy_diff',
                        help='Fairness metric, has to be in ["spd", "equalized_odds", "accuracy_diff"]. '
                             'Note: with --intersectional only "spd" is supported.')

    parser.add_argument('--method', type=str,
                        default='all',
                        help='Which method you would like Pareto fronts for',
                        choices=["two_trees",
                                 "weighted_combi",
                                 "constrained",
                                 "backtracking",
                                 "postprocessing",
                                 "chebyshev",
                                 "gp_nsga2",
                                 "gp_sms",
                                 "all"
                                 ])

    args = parser.parse_args()

    os.makedirs(os.path.join(PROJECT_ROOT, args.path_output, 'plots_' + args.fairness_metric + "/"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, args.path_output, 'evaluation_' + args.fairness_metric + "/"), exist_ok=True)

    methods = ["Two trees linear",
               "Two trees meta tree",
               "Two trees meta tree opt",
               "One tree (combined split criterion)",
               "One tree (constrained split criterion)",
               "One tree (backtracking)",
               # "Threshold optimizer",
               "One Tree (chebyshev)",
               "GP tree (NSGA-II)",
               "GP tree (SMS-EMOA)"
               ]

    file_names = ["two_trees_linear",
                  "two_trees_mt",
                  "two_trees_mto",
                  "weighted_combi",
                  "constrained",
                  "backtracking",
                  # "postprocessing",
                  "chebyshev",
                  "gp_nsga2",
                  "gp_sms"
                  ]

    method_directories = ["tradeoffs_two_tree/linear_AND_fairness_gain_AND_own/" + args.fairness_metric,
                          "tradeoffs_two_tree/meta_tree_AND_fairness_gain_AND_own/" + args.fairness_metric,
                          "tradeoffs_two_tree/meta_tree_optimization_AND_fairness_gain_AND_own/" + args.fairness_metric,
                          "tradeoffs_one_tree/weighted_combi/" + args.fairness_metric,
                          "tradeoffs_one_tree/threshold_gain_s/" + args.fairness_metric,
                          "tradeoffs_one_tree/backtracking/" + args.fairness_metric,
                          # "relaxed_threshold_optimizer",
                          "tradeoffs_one_tree/chebyshev/" + args.fairness_metric,
                          "tradeoffs_gp_tree/nsga2",
                          "tradeoffs_gp_tree/sms"
                          ]

    if args.method != 'all':
        methods = [methods[file_names.index(args.method)]]
        method_directories = [method_directories[file_names.index(args.method)]]
        file_names = [file_names[file_names.index(args.method)]]

    directory = os.path.join(PROJECT_ROOT, args.path_input, args.data)
    # print("test HV", hypervolume(pd.DataFrame([0, 0.5, 1]), pd.DataFrame([0, 0.5, 1])))

    results = {
        "method": [],
        "avg_autoc": [],
        "avg_autoc_pareto": [],
        "avg_num_pareto_points_local": [],
        "avg_num_unique_paretos": [],
        "avg_num_unique_points": [],
        "avg_hypervolume": [],
        "avg_gd": [],
        "avg_gdplus": [],
        "avg_igd": [],
        "avg_igdplus": [],
        "avg_spread": [],
        "avg_distr_variance": [],
        "std_autoc": [],
        "std_autoc_pareto": [],
        "std_num_pareto_points_local": [],
        "std_num_unique_paretos": [],
        "std_num_unique_points": [],
        "std_distr_variance": [],
        "std_hypervolume": [],
        "std_gd": [],
        "std_gdplus": [],
        "std_igd": [],
        "std_igdplus": [],
        "std_spread": [],
    }

    for i, method in enumerate(methods):
        method_full_path = os.path.join(directory, method_directories[i])
        if not os.path.exists(method_full_path):
            continue
        results["method"].append(method)
        avg_curve, results = get_avg_dataframe(method_full_path, results)
        avg_curve["1-spds_test"] = 1 - avg_curve["spds_test"]
        # plt.scatter(avg_curve["aurocs_test"], avg_curve["1-spds_test"], label=method)
        pd.DataFrame(avg_curve).to_csv(os.path.join(PROJECT_ROOT, args.path_output, 'plots_' + args.fairness_metric, "avg_curve_" + args.data + "_" + file_names[i] + ".csv"))

        #plot_all_best_curves(method_full_path, avg_curve)

    # plt.legend()
    # plt.show()
    pd.DataFrame(results).to_csv(os.path.join(PROJECT_ROOT, args.path_output, 'evaluation_' + args.fairness_metric, args.data + ".csv"))


def get_avg_dataframe(directory, results):
    csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv') and file.startswith("best_results")]
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file)
        if "gamma" in df.columns and "gammas" not in df.columns:
            df = df.rename(columns={"gamma": "gammas"})
        dataframes.append(df)

    for df in dataframes:
        df.loc[df['aurocs_test'] < 0.5, 'aurocs_test'] = 1 - df['aurocs_test']

    autocs = [autoc(df['spds_test'], df['aurocs_test']) for df in dataframes]

    distr_vars = [check_point_distribution(df['spds_test'], df['aurocs_test']) for df in dataframes]
    unique_points_list = [num_unique_points(df['spds_test'], df['aurocs_test']) for df in dataframes]
    pareto_results = [num_local_pareto_points(df['spds_test'], df['aurocs_test']) for df in dataframes]
    local_paretos_list = [pareto_results[i][0] for i in range(len(dataframes))]
    unique_paretos_list = [pareto_results[i][1] for i in range(len(dataframes))]
    hypervolumes = [hypervolume(pareto_results[i][2]['1-spds_test'], pareto_results[i][2]['aurocs_test']) for i in range(len(dataframes))]
    autocs_pareto = [autoc(1 - pareto_results[i][2]['1-spds_test'], pareto_results[i][2]['aurocs_test']) for i in range(len(dataframes))]
    distances = [general_distance(
        dataframes[i]['spds_test'], dataframes[i]['aurocs_test'],
        np.column_stack([- pareto_results[i][2]['aurocs_test'], - pareto_results[i][2]['1-spds_test']])) for i in range(len(dataframes))]

    gd, gd_plus, igd, igd_plus = map(list, zip(*distances))
    spread = [deb_spread(
        dataframes[i]['spds_test'], dataframes[i]['aurocs_test'],
        np.column_stack([- pareto_results[i][2]['aurocs_test'], - pareto_results[i][2]['1-spds_test']])) for i in range(len(dataframes))]

    results["avg_autoc"].append(round(np.mean(autocs), 4))
    results["avg_autoc_pareto"].append(round(np.mean(autocs_pareto), 4))
    results["avg_num_pareto_points_local"].append(round(np.mean(local_paretos_list), 4))
    results["avg_num_unique_paretos"].append(round(np.mean(unique_paretos_list), 4))
    results["avg_num_unique_points"].append(round(np.mean(unique_points_list), 4))
    results["avg_distr_variance"].append(round(np.mean(distr_vars), 4))
    results["avg_hypervolume"].append(round(np.mean(hypervolumes), 4))
    results["avg_gd"].append(round(np.mean(gd), 4))
    results["avg_gdplus"].append(round(np.mean(gd_plus), 4))
    results["avg_igd"].append(round(np.mean(igd), 4))
    results["avg_igdplus"].append(round(np.mean(igd_plus), 4))
    results["avg_spread"].append(round(np.mean(spread), 4))

    results["std_autoc"].append(round(np.std(autocs), 4))
    results["std_autoc_pareto"].append(round(np.std(autocs_pareto), 4))
    results["std_num_pareto_points_local"].append(round(np.std(local_paretos_list), 4))
    results["std_num_unique_paretos"].append(round(np.std(unique_paretos_list), 4))
    results["std_num_unique_points"].append(round(np.std(unique_points_list), 4))
    results["std_distr_variance"].append(round(np.std(distr_vars), 4))
    results["std_hypervolume"].append(round(np.std(hypervolumes), 4))
    results["std_gd"].append(round(np.std(gd), 4))
    results["std_gdplus"].append(round(np.std(gd_plus), 4))
    results["std_igd"].append(round(np.std(igd), 4))
    results["std_igdplus"].append(round(np.std(igd_plus), 4))
    results["std_spread"].append(round(np.std(spread), 4))

    # Calculate averaged dataframe
    combined_df = pd.concat(dataframes)
    if 'gammas' in combined_df.columns:
        # gamma-parameterised methods: seeds align by gamma, so average per gamma value
        averaged_df = combined_df.groupby('gammas')[['spds_test', 'aurocs_test', 'spds_train', 'aurocs_train']].mean().reset_index()
    else:
        # methods without a gamma parameterisation (e.g. GP structural MOO): fronts are not
        # aligned across seeds, so pool all seeds and keep the non-dominated (Pareto) points
        averaged_df = pooled_pareto_curve(combined_df)
    return averaged_df, results


def pooled_pareto_curve(df):
    """
    Summary curve for methods whose per-seed fronts are not aligned by a shared parameter:
    pool every seed's points and return the non-dominated (Pareto-optimal) subset.

    Dominance is evaluated on (aurocs_test, 1 - spds_test) with both higher-is-better,
    matching ``num_local_pareto_points``. Train columns (if present) are carried along.
    """
    df = df.reset_index(drop=True)
    a = df['aurocs_test'].to_numpy()
    f = 1.0 - df['spds_test'].to_numpy()  # higher is better
    keep = []
    for i in range(len(df)):
        dominated = False
        for j in range(len(df)):
            if j != i and a[j] > a[i] and f[j] > f[i]:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    cols = [c for c in ['spds_test', 'aurocs_test', 'spds_train', 'aurocs_train'] if c in df.columns]
    return df.iloc[keep][cols].drop_duplicates().reset_index(drop=True)


def plot_all_best_curves(directory, avg_dataframe):
    csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]
    dataframes = [pd.read_csv(file) for file in csv_files if os.path.basename(file).startswith("best_results")]
    # plot
    fig, ax = plt.subplots(8, 2, figsize=(5, 10))
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()

    size = 12

    for i, df in enumerate(dataframes):
        row = i % 8  # Row index resets every 8 plots
        col = (i // 8)  # Move to the next column after 8 plots

        ax[row, col].grid(color='lightgrey', linestyle='-', linewidth=0.5, alpha=0.4)
        ax[row, col].scatter(df["aurocs_test"], [1 - d for d in df["spds_test"]], s=3, label="test")
        ax[row, col].scatter(df["aurocs_train"], [1 - d for d in df["spds_train"]], s=3, label="train")
        ax[row, col].set_xlabel("auroc")
    ax[7, 1].scatter(avg_dataframe["aurocs_test"], [1 - d for d in avg_dataframe["spds_test"]],
                     label="Two trees", color="red", s=size, marker='o')
    ax[0, 0].legend()
    plt.show()


if __name__ == "__main__":
    main()

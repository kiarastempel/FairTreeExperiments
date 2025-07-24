import matplotlib.pyplot as plt
import argparse
from algos_two_trees.get_data import *
import glob
import numpy as np
import pandas as pd
from evaluation.evaluation_metrics import autoc, num_unique_points, \
    check_point_distribution, num_local_pareto_points

seed = 42


def main():
    parser = argparse.ArgumentParser(description='Get run values.')

    parser.add_argument('--data', type=str, default='Adult',
                        help='Dataset to use: Compas, Adult, Banks, German, Law, Dutch')

    parser.add_argument('--path_input', type=str, default='results_best/',
                        help='Path to specific input results folder')

    parser.add_argument('--path_output', type=str,
                        default='results_best/',
                        help='Path to specific folder where the output will be saved')

    args = parser.parse_args()

    os.makedirs(os.path.join(PROJECT_ROOT, args.path_output, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, args.path_output, 'evaluation'), exist_ok=True)

    methods = ["Two trees",
               "One tree (combined split criterion)",
               "One tree (constrained split criterion)",
               "One tree (backtracking)",
               "Threshold optimizer"
               ]

    file_names = ["two_trees",
                  "weighted_combi",
                  "constrained",
                  "backtracking",
                  "postprocessing"
                  ]

    method_directories = ["tradeoffs_two_trees/fairness_gain_AND_sklearn",
                          "tradeoffs_one_tree/weighted_combi",
                          "tradeoffs_one_tree/threshold_gain_s",
                          "tradeoffs_one_tree/backtracking",
                          "relaxed_threshold_optimizer"
                          ]

    directory = os.path.join(PROJECT_ROOT, args.path_input, args.data)

    results = {
        "method": [],
        "avg_autoc": [],
        "num_pareto_points_local": [],
        "num_unique_paretos": [],
        "num_unique_points": [],
        "distr_variance": [],
        "std_autoc": [],
        "std_pareto_points_local": [],
        "std_num_unique_paretos": [],
        "std_num_unique_points": [],
        "std_distr_variance": []
    }

    for i, method in enumerate(methods):
        method_full_path = os.path.join(directory, method_directories[i])
        avg_curve, avg_autoc, avg_distr_var, avg_unique_points, avg_num_local_paretos, avg_unique_paretos, \
            std_autoc, std_distr_var, std_unique_points, std_num_local_paretos, std_unique_paretos = \
            get_avg_dataframe(method_full_path)
        avg_curve["1-spds_test"] = 1 - avg_curve["spds_test"]
        plt.scatter(avg_curve["aurocs_test"], avg_curve["1-spds_test"], label=method)
        pd.DataFrame(avg_curve).to_csv(os.path.join(PROJECT_ROOT, args.path_output, 'plots', "avg_curve_" + args.data + "_" + file_names[i] + ".csv"))
        results["method"].append(method)
        results["avg_autoc"].append(round(avg_autoc, 4))
        results["num_pareto_points_local"].append(round(avg_num_local_paretos, 4))
        results["num_unique_paretos"].append(round(avg_unique_paretos, 4))
        results["num_unique_points"].append(round(avg_unique_points, 4))
        results["distr_variance"].append(round(avg_distr_var, 4))
        results["std_autoc"].append(round(std_autoc, 4))
        results["std_pareto_points_local"].append(round(std_num_local_paretos, 4))
        results["std_num_unique_paretos"].append(round(std_unique_paretos, 4))
        results["std_num_unique_points"].append(round(std_unique_points, 4))
        results["std_distr_variance"].append(round(std_distr_var, 4))

        # plot_all_best_curves(method_full_path, avg_curve)

    plt.legend()
    plt.show()
    pd.DataFrame(results).to_csv(os.path.join(PROJECT_ROOT, args.path_output, 'evaluation', args.data + ".csv"))


def get_avg_dataframe(directory):
    csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv') and file.startswith("best_results")]
    dataframes = [pd.read_csv(file) for file in csv_files]

    for df in dataframes:
        df.loc[df['aurocs_test'] < 0.5, 'aurocs_test'] = 1 - df['aurocs_test']

    autocs = [autoc(df['spds_test'], df['aurocs_test']) for df in dataframes]
    distr_vars = [check_point_distribution(df['spds_test'], df['aurocs_test']) for df in dataframes]
    unique_points_list = [num_unique_points(df['spds_test'], df['aurocs_test']) for df in dataframes]
    local_paretos_list = [num_local_pareto_points(df['spds_test'], df['aurocs_test'])[0] for df in dataframes]
    unique_paretos_list = [num_local_pareto_points(df['spds_test'], df['aurocs_test'])[1] for df in dataframes]

    avg_autoc = np.mean(autocs)
    avg_distr_var = np.mean(distr_vars)
    avg_unique_points = np.mean(unique_points_list)
    avg_num_local_paretos = np.mean(local_paretos_list)
    avg_unique_paretos = np.mean(unique_paretos_list)

    std_autoc = np.std(autocs)
    std_distr_var = np.std(distr_vars)
    std_unique_points = np.std(unique_points_list)
    std_num_local_paretos = np.std(local_paretos_list)
    std_unique_paretos = np.std(unique_paretos_list)

    # Calculate averaged dataframe
    combined_df = pd.concat(dataframes)
    averaged_df = combined_df.groupby('gammas')[['spds_test', 'aurocs_test', 'spds_train', 'aurocs_train']].mean().reset_index()
    return averaged_df, avg_autoc, avg_distr_var, avg_unique_points, avg_num_local_paretos, avg_unique_paretos, std_autoc, std_distr_var, std_unique_points, std_num_local_paretos, std_unique_paretos


def plot_all_best_curves(directory, avg_dataframe):
    csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]
    print(csv_files)
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

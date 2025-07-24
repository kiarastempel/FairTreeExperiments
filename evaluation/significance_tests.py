import argparse
import os
from constants import PROJECT_ROOT
import numpy as np
import pandas as pd
from evaluation.evaluation_metrics import autoc
from scipy.stats import ttest_ind


seed = 42


def main():
    parser = argparse.ArgumentParser(description='Get run values.')

    parser.add_argument('--path_a', type=str, default='results_best_flipped/Adult/tradeoffs_one_tree/weighted_combi')

    parser.add_argument('--path_b', type=str,
                        default='results_best_flipped/Adult/tradeoffs_one_tree/threshold_gain_s')

    args = parser.parse_args()

    a = get_autocs(args.path_a)
    b = get_autocs(args.path_b)

    tt = ttest_ind(a, b, random_state=seed)

    print("p_value: ", tt.pvalue)


def get_autocs(path):
    method_full_path = os.path.join(PROJECT_ROOT, path)
    csv_files = [os.path.join(method_full_path, file) for file in os.listdir(method_full_path) if file.endswith('.csv') and file.startswith("best_results")]
    dataframes = [pd.read_csv(file) for file in csv_files]

    for df in dataframes:
        df.loc[df['aurocs_test'] < 0.5, 'aurocs_test'] = 1 - df['aurocs_test']
    autocs = [autoc(df['spds_test'], df['aurocs_test']) for df in dataframes]

    return autocs


if __name__ == "__main__":
    main()

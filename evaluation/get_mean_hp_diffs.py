import os
import argparse
import pandas as pd
from constants import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str,
                        default='results_best_MLJ_old_opt_hypervolume/hypervolume_difference.csv',
                        help='Path to hypervolume_difference.csv')
    parser.add_argument('--output_path', type=str,
                        default='results_best_MLJ_old_opt_hypervolume',
                        help='Where to save aggregated results')

    args = parser.parse_args()

    input_path = os.path.join(PROJECT_ROOT, args.input_file)
    output_dir = os.path.join(PROJECT_ROOT, args.output_path)

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)

    # Aggregation per method
    agg = (
        df.groupby("method")["diff"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    agg = agg.rename(columns={
        "mean": "mean_diff",
        "std": "std_diff",
        "count": "num_datasets"
    })

    # round for readability
    agg["mean_diff"] = agg["mean_diff"].round(6)
    agg["std_diff"] = agg["std_diff"].round(6)

    agg.to_csv(
        os.path.join(output_dir, "hypervolume_difference_aggregated.csv"),
        index=False
    )

    print("Done. Saved hypervolume_difference_aggregated.csv")


if __name__ == "__main__":
    main()

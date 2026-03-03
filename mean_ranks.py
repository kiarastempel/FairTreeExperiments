import os
import pandas as pd

MAXIMIZE = {
    "hypervolume",
    "distr_variance",
    "num_pareto_points_local",
    "num_unique_paretos",
    "num_unique_points",
}

MINIMIZE = {
    "gd",
    "gdplus",
    "igd",
    "igdplus",
    "spread",
}

# Only these columns in this order:
LATEX_COLUMNS = [
    "avg_hypervolume",
    "avg_distr_variance",
    "avg_gd",
    "avg_igdplus",
    "avg_num_unique_paretos",
]
# LATEX_COLUMNS = [
#     "avg_gdplus",
#     "avg_igd",
#     "avg_num_pareto_points_local",
#     "avg_num_unique_points",
#     "avg_spread",
# ]

VALID_METRICS = MAXIMIZE.union(MINIMIZE)


def compute_mean_ranks(directory_path):
    dataset_files = sorted([
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.endswith(".csv")
    ])

    all_ranks = []
    method_order = None

    for i, file in enumerate(dataset_files):
        df = pd.read_csv(file)

        if i == 0:
            method_order = df["method"].tolist()

        avg_cols = [
            col for col in df.columns
            if col.startswith("avg_") and col.replace("avg_", "") in VALID_METRICS
        ]

        df = df[["method"] + avg_cols].copy()
        df_ranks = df.copy()

        for col in avg_cols:
            metric_name = col.replace("avg_", "")

            if metric_name in MAXIMIZE:
                ascending = False
            else:
                ascending = True

            df_ranks[col] = df[col].rank(
                ascending=ascending,
                method="average"
            )

        all_ranks.append(df_ranks)

    combined = pd.concat(all_ranks)

    mean_ranks = (
        combined
        .groupby("method", sort=False)
        .mean()
        .reset_index()
    )

    # Restore original order
    mean_ranks["method"] = pd.Categorical(
        mean_ranks["method"],
        categories=method_order,
        ordered=True
    )
    mean_ranks = mean_ranks.sort_values("method").reset_index(drop=True)

    # Keep only selected columns
    mean_ranks = mean_ranks[["method"] + LATEX_COLUMNS]

    # Round
    mean_ranks = mean_ranks.round(2)

    # ---------- SAVE CSV ----------
    csv_path = os.path.join(directory_path, "mean_ranks.csv")
    mean_ranks.to_csv(csv_path, index=False)

    # ---------- CREATE LATEX TABLE ----------
    latex_lines = []

    for i, row in mean_ranks.iterrows():
        values = [str(row[col]) for col in LATEX_COLUMNS]

        if i == 0:
            prefix = "Mean rank"
        else:
            prefix = ""

        latex_lines.append(
            f"{prefix} & {row['method']} & " + " & ".join(values) + " \\\\"
        )

    latex_code = "\n".join(latex_lines)

    txt_path = os.path.join(directory_path, "mean_ranks_table.txt")
    with open(txt_path, "w") as f:
        f.write(latex_code)

    print(f"Mean ranks saved to: {csv_path}")
    print(f"LaTeX table saved to: {txt_path}")


# Example:
compute_mean_ranks("results_best_overall")

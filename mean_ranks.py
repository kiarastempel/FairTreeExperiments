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

# The two column variants. "1" = the set that used to be uncommented,
# "2" = the set that used to be the comment.
COLUMN_VARIANTS = {
    "1": [
        "avg_hypervolume",
        "avg_distr_variance",
        "avg_gd",
        "avg_igdplus",
        "avg_num_unique_paretos",
    ],
    "2": [
        "avg_gdplus",
        "avg_igd",
        "avg_num_pareto_points_local",
        "avg_num_unique_points",
        "avg_spread",
    ],
}

# CSV method name -> label used in the LaTeX table
METHOD_NAME_MAP = {
    "Two trees linear": "2TFT",
    "Two trees meta tree": "2TFT-M",
    "Two trees meta tree opt": "2TFT-M (hp opt.)",
    "One tree (combined split criterion)": "One tree (combined SP)",
    "One tree (constrained split criterion)": "One tree (constrained SP)",
    "One tree (backtracking)": "DTFC (backtracking)",
    "One Tree (chebyshev)": "One tree (Chebyshev)",
    "NSGA-II (constrained split criterion)": "NSGA-II (constrained SP)",
    "SMS-EMOA (constrained split criterion)": "SMS-EMOA (constrained SP)",
    "GP tree (NSGA-II)": "NSGA-II (GP)",
    "GP tree (SMS-EMOA)": "SMS-EMOA (GP)",
}

VALID_METRICS = MAXIMIZE.union(MINIMIZE)


def _compute_mean_ranks_df(directory_path):
    """Rank every method per metric per dataset, then average ranks across datasets."""
    dataset_files = sorted([
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.endswith(".csv") and not f.startswith("mean_ranks")
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
            ascending = metric_name not in MAXIMIZE  # MAXIMIZE -> rank 1 = highest
            df_ranks[col] = df[col].rank(ascending=ascending, method="average")

        all_ranks.append(df_ranks)

    combined = pd.concat(all_ranks)
    mean_ranks = combined.groupby("method", sort=False).mean().reset_index()

    # restore original method order
    mean_ranks["method"] = pd.Categorical(
        mean_ranks["method"], categories=method_order, ordered=True
    )
    mean_ranks = mean_ranks.sort_values("method").reset_index(drop=True)
    return mean_ranks


def _write_variant(mean_ranks, directory_path, columns, suffix):
    """Write csv + LaTeX txt for one column variant."""
    table = mean_ranks[["method"] + columns].copy().round(2)

    csv_path = os.path.join(directory_path, f"mean_ranks_{suffix}.csv")
    table.to_csv(csv_path, index=False)

    latex_lines = []
    for i, row in table.iterrows():
        values = [str(row[col]) for col in columns]
        prefix = "Mean rank" if i == 0 else ""
        label = METHOD_NAME_MAP.get(row["method"], row["method"])
        latex_lines.append(f"{prefix} & {label} & " + " & ".join(values) + r" \\")

    txt_path = os.path.join(directory_path, f"mean_ranks_table_{suffix}.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(latex_lines))

    print(f"[variant {suffix}] CSV  -> {csv_path}")
    print(f"[variant {suffix}] LaTeX -> {txt_path}")


def compute_mean_ranks(directory_path):
    mean_ranks = _compute_mean_ranks_df(directory_path)
    for suffix, columns in COLUMN_VARIANTS.items():
        _write_variant(mean_ranks, directory_path, columns, suffix)


# Example:
compute_mean_ranks("results_best_overall_spd")


import os
import argparse
import numpy as np
import pandas as pd

from constants import PROJECT_ROOT
from evaluation_metrics import hypervolume


def compute_pareto(points):
    pareto = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i != j:
                if (q[0] >= p[0] and q[1] >= p[1]) and (q[0] > p[0] or q[1] > p[1]):
                    dominated = True
                    break
        if not dominated:
            pareto.append(p)
    return np.array(pareto)


def collect_method_seed_points(dataset_path):
    method_seed_points = {}

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.startswith("best_results") and file.endswith(".csv"):
                filepath = os.path.join(root, file)

                method = root  # unique per subfolder

                seed = None
                parts = file.split("_")
                for i, p in enumerate(parts):
                    if p == "seed":
                        seed = parts[i + 1].replace(".csv", "")
                        break
                if seed is None:
                    continue

                df = pd.read_csv(filepath)
                df.loc[df['aurocs_test'] < 0.5, 'aurocs_test'] = 1 - df['aurocs_test']

                points = np.column_stack((
                    1 - df['spds_test'],
                    df['aurocs_test']
                ))

                if method not in method_seed_points:
                    method_seed_points[method] = {}

                method_seed_points[method][seed] = points

    return method_seed_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_input', type=str,
                        default='results_best_MLJ_old_opt_hypervolume',
                        help='Root folder containing dataset subfolders')
    parser.add_argument('--path_output', type=str,
                        default='results_best_MLJ_old_opt_hypervolume',
                        help='Where to save the diff csv')

    args = parser.parse_args()

    input_root = os.path.join(PROJECT_ROOT, args.path_input)
    output_root = os.path.join(PROJECT_ROOT, args.path_output)

    os.makedirs(output_root, exist_ok=True)

    final_rows = []

    for dataset in os.listdir(input_root):

        dataset_path = os.path.join(input_root, dataset)
        if not os.path.isdir(dataset_path):
            continue

        method_seed_points = collect_method_seed_points(dataset_path)

        # determine all seeds
        all_seeds = set()
        for m in method_seed_points:
            all_seeds.update(method_seed_points[m].keys())
        all_seeds = sorted(list(all_seeds))

        # --- compute overall HV ---
        overall_seed_hv = []

        for seed in all_seeds:
            combined_points = []

            for method in method_seed_points:
                seeds_available = method_seed_points[method]

                if seed in seeds_available:
                    combined_points.append(seeds_available[seed])
                else:
                    # replicate if only one seed exists
                    if len(seeds_available) == 1:
                        only_seed = list(seeds_available.keys())[0]
                        combined_points.append(seeds_available[only_seed])

            combined_points = np.vstack(combined_points)
            pareto = compute_pareto(combined_points)
            hv = hypervolume(pareto[:, 0], pareto[:, 1])
            overall_seed_hv.append(hv)

        overall_mean = np.mean(overall_seed_hv)

        # --- compute method HV ---
        for method in method_seed_points:

            method_seed_hv = []

            for seed in all_seeds:
                seeds_available = method_seed_points[method]

                if seed in seeds_available:
                    points = seeds_available[seed]
                else:
                    if len(seeds_available) == 1:
                        only_seed = list(seeds_available.keys())[0]
                        points = seeds_available[only_seed]
                    else:
                        continue

                pareto = compute_pareto(points)
                hv = hypervolume(pareto[:, 0], pareto[:, 1])
                method_seed_hv.append(hv)

            if len(method_seed_hv) == 0:
                continue

            method_mean = np.mean(method_seed_hv)

            final_rows.append({
                "dataset": dataset,
                "method": os.path.basename(method),
                "overall_hypervolume": round(overall_mean, 6),
                "hypervolume_method": round(method_mean, 6),
                "diff": round(method_mean - overall_mean, 6)
            })

    df_final = pd.DataFrame(final_rows)
    df_final.to_csv(
        os.path.join(output_root, "hypervolume_difference.csv"),
        index=False
    )

    print("Done. Saved hypervolume_difference.csv")


if __name__ == "__main__":
    main()

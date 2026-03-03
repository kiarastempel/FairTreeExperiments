import os
import argparse
import numpy as np
import pandas as pd

from constants import PROJECT_ROOT
from evaluation_metrics import hypervolume


def collect_method_seed_points(dataset_path):
    """
    Collect points per method per seed.
    Returns:
        method_seed_points[method][seed] = np.array(points)
    """
    method_seed_points = {}

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.startswith("best_results") and file.endswith(".csv"):
                filepath = os.path.join(root, file)

                method = root  # unique identifier via folder path

                # extract seed
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_input', type=str, default='results_best_MLJ_old_opt_hypervolume',
                        help='Path to specific input results folder')

    parser.add_argument('--path_output', type=str, default='results_best_MLJ_old_opt_hypervolume',
                        help='Path to specific folder where the output will be saved')
    args = parser.parse_args()

    input_root = os.path.join(PROJECT_ROOT, args.path_input)
    output_root = os.path.join(PROJECT_ROOT, args.path_output)
    os.makedirs(output_root, exist_ok=True)

    dataset_results = []

    for dataset in os.listdir(input_root):
        dataset_path = os.path.join(input_root, dataset)
        if not os.path.isdir(dataset_path):
            continue

        method_seed_points = collect_method_seed_points(dataset_path)

        # Determine global seed set
        all_seeds = set()
        for method in method_seed_points:
            all_seeds.update(method_seed_points[method].keys())

        all_seeds = sorted(list(all_seeds))

        seed_hvs = []

        for seed in all_seeds:
            combined_points = []

            for method in method_seed_points:
                seeds_available = method_seed_points[method]

                if seed in seeds_available:
                    combined_points.append(seeds_available[seed])
                else:
                    # replicate single-seed method
                    if len(seeds_available) == 1:
                        only_seed = list(seeds_available.keys())[0]
                        combined_points.append(seeds_available[only_seed])
                    # if method has multiple seeds but not this one,
                    # we ignore (should not normally happen)

            if len(combined_points) == 0:
                continue

            combined_points = np.vstack(combined_points)

            pareto_front = compute_pareto(combined_points)
            hv = hypervolume(pareto_front[:, 0], pareto_front[:, 1])

            seed_hvs.append(hv)

        if len(seed_hvs) == 0:
            continue

        dataset_results.append({
            "dataset": dataset,
            "mean_hypervolume": round(np.mean(seed_hvs), 6),
            "std_hypervolume": round(np.std(seed_hvs), 6),
            "num_seeds": len(seed_hvs)
        })

    df_results = pd.DataFrame(dataset_results)
    df_results.to_csv(
        os.path.join(output_root, "overall_hypervolume_per_dataset.csv"),
        index=False
    )

    print("Done.")


if __name__ == "__main__":
    main()
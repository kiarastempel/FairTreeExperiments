import pandas as pd
import numpy as np
import argparse
from scipy.spatial.distance import pdist


def autoc(spds, aurocs):
    # transform to 1 - spds (y-axis)
    one_minus_spds_test = 1 - spds

    # combine the data points and sort them by `aurocs_test` (x-axis)
    curve_points = pd.DataFrame({'aurocs_test': aurocs, '1_minus_spds_test': one_minus_spds_test}).sort_values(
        by='aurocs_test')

    curve_points = curve_points[curve_points['aurocs_test'] > 0.5]

    if len(curve_points) == 0:
        return 0

    first_point = pd.DataFrame({'aurocs_test': [0.5], '1_minus_spds_test': [max(curve_points['1_minus_spds_test'])]})
    last_point = pd.DataFrame({'aurocs_test': [1], '1_minus_spds_test': [min(curve_points['1_minus_spds_test'])]})

    curve_points = pd.concat([first_point, curve_points, last_point], ignore_index=True)

    curve_points = curve_points.sort_values(by='aurocs_test').reset_index(drop=True)

    # Calculate the area under the curve using the trapezoidal rule
    auc = round(np.trapezoid(curve_points['1_minus_spds_test'], curve_points['aurocs_test']), 4)
    return auc


def check_point_distribution(spds, aurocs):
    # transform to 1 - spds (y-axis)
    one_minus_spds_test = 1 - spds

    # combine the data points and sort them by `aurocs_test` (x-axis)
    curve_points = pd.DataFrame({'aurocs_test': aurocs, '1_minus_spds_test': one_minus_spds_test}).sort_values(
        by='aurocs_test')
    curve_points = curve_points[curve_points['aurocs_test'] > 0.5]

    distances = pdist(curve_points[['1_minus_spds_test', 'aurocs_test']])

    if len(distances) == 0:
        return 0

    # Compute variance of these distances
    distribution_score = distances.var()
    return distribution_score


def num_unique_points(spds, aurocs):
    # transform to 1 - spds (y-axis)
    one_minus_spds_test = 1 - spds

    # combine the data points and sort them by `aurocs_test` (x-axis)
    curve_points = pd.DataFrame({'aurocs_test': aurocs, '1_minus_spds_test': one_minus_spds_test}).sort_values(by='aurocs_test')
    unique_points_count = curve_points[['1_minus_spds_test', 'aurocs_test']].drop_duplicates().shape[0]
    return unique_points_count


def num_local_pareto_points(spds, aurocs):
    points = list(zip(aurocs, 1 - spds))  # Combine into (x, y) pairs
    num = 0
    pareto_points = {"1-spds_test": [], "aurocs_test": []}

    for i, (x_i, y_i) in enumerate(points):
        dominated = False
        for j, (x_j, y_j) in enumerate(points):
            if j != i:
                # Check if point j dominates point i
                if x_j > x_i and y_j > y_i:
                    dominated = True
                    pareto_points["1-spds_test"].append(x_i)
                    pareto_points["aurocs_test"].append(y_i)
                    break  # No need to check further if already dominated
        if not dominated:
            num += 1  # Point i is Pareto-optimal
    # calculate unique Pareto-optimal points
    num_uniques = pd.DataFrame(pareto_points)[['1-spds_test', 'aurocs_test']].drop_duplicates().shape[0]
    return num, num_uniques


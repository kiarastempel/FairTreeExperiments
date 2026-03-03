import pandas as pd
import numpy as np
import argparse
from scipy.spatial.distance import pdist
from pymoo.indicators.hv import HV
from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus


def hypervolume(spds, aurocs, ref_point=(0.0, 0.0)):
    """Compute hypervolume for fairness-performance trade-off"""
    one_minus_spds = np.asarray(spds)
    aurocs = np.asarray(aurocs)

    f = np.column_stack([
        -aurocs,
        -one_minus_spds
    ])

    ind = HV(ref_point=ref_point)
    return ind(f)


def general_distance(spds, aurocs, pareto_front):
    """Calculate:
    (1) Average distance from any point of the trade-off curve to the closest point in the Pareto-front (euclidean distance).
    (2) As (1), but with modified distance. For minimization: d_i = max{a_i - z_i, 0} for a_i being the solution,
    z_i the closest reference point, so that only distances where a is worse are penalized, not if a has a high distance
    only because it is better than the reference point on the Pareto front.
    (3) Analogous to (1), but instead we measure (inverted) the distance from any point of the pareto front to the closest point of
    the trade-off curve.
    (4) As (3), but adapted as in (2)."""
    one_minus_spds = np.asarray(1 - spds)
    aurocs = np.asarray(aurocs)

    f = np.column_stack([
        -aurocs,
        -one_minus_spds
    ])

    ind_gd = GD(pareto_front)
    ind_gdplus = GDPlus(pareto_front)
    ind_igd = IGD(pareto_front)
    ind_igdplus = IGDPlus(pareto_front)
    return ind_gd(f), ind_gdplus(f), ind_igd(f), ind_igdplus(f)


def deb_spread(spds, aurocs, pareto_front):
    """
    Compute Deb et al.'s spread metric. Lower is better. Only works for two objectives.
    """
    spds = np.asarray(spds).reshape(-1)
    aurocs = np.asarray(aurocs).reshape(-1)

    f = np.column_stack([
        -aurocs,
        -(1.0 - spds)
    ])

    # we need at least 2 points
    if f.shape[0] < 2:
        return 0.0

    # sort approximation front (=trade-off curve) by first objective
    f = f[np.argsort(f[:, 0])]

    # compute extreme points from Pareto front
    # extreme for objective 1
    extreme_1 = pareto_front[np.argmin(pareto_front[:, 0])]
    # extreme for objective 2
    extreme_2 = pareto_front[np.argmin(pareto_front[:, 1])]

    # compute df and dl
    d_f = np.linalg.norm(f[0] - extreme_1)
    d_l = np.linalg.norm(f[-1] - extreme_2)

    # distances between consecutive points in trade-off curve
    dists = np.linalg.norm(f[1:] - f[:-1], axis=1)
    d_bar = np.mean(dists)

    # Deb spread formula
    delta = (d_f + d_l + np.sum(np.abs(dists - d_bar))) / (
        d_f + d_l + (len(dists)) * d_bar
    )

    return delta


def autoc(spds, aurocs, ref_point=(0, 0)):  # previously (0.5,  0)
    # transform to 1 - spds (y-axis)
    one_minus_spds_test = 1 - spds

    # combine the data points and sort them by `aurocs_test` (x-axis)
    curve_points = pd.DataFrame({'aurocs_test': aurocs, '1_minus_spds_test': one_minus_spds_test}).sort_values(
        by='aurocs_test')

    curve_points = curve_points[curve_points['aurocs_test'] > 0.5]

    if len(curve_points) == 0:
        return 0

    # first_point = pd.DataFrame({'aurocs_test': [0.5], '1_minus_spds_test': [max(curve_points['1_minus_spds_test'])]})
    first_point = pd.DataFrame({'aurocs_test': [ref_point[0]], '1_minus_spds_test': [max(curve_points['1_minus_spds_test'])]})
    # last_point = pd.DataFrame({'aurocs_test': [1], '1_minus_spds_test': [min(curve_points['1_minus_spds_test'])]})
    last_point = pd.DataFrame({'aurocs_test': [max(curve_points['aurocs_test'])], '1_minus_spds_test': [ref_point[0]]})

    curve_points = pd.concat([first_point, curve_points, last_point], ignore_index=True)

    curve_points = curve_points.sort_values(by='aurocs_test').reset_index(drop=True)

    # Calculate the area under the curve using the trapezoidal rule
    auc = round(np.trapezoid(curve_points['1_minus_spds_test'], curve_points['aurocs_test']), 4)
    return auc


def check_point_distribution(spds, aurocs):
    """Calculate the variance of pairwise distances of all points in the trade-off curve.

    Corresponds to the spacing metric (MOO)."""
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
                    break  # No need to check further if already dominated
        if not dominated:
            num += 1  # Point i is Pareto-optimal
            pareto_points["1-spds_test"].append(x_i)
            pareto_points["aurocs_test"].append(y_i)
    # calculate unique Pareto-optimal points
    pareto_front = pd.DataFrame(pareto_points)[['1-spds_test', 'aurocs_test']].drop_duplicates()
    num_uniques = pareto_front.shape[0]
    return num, num_uniques, pareto_front



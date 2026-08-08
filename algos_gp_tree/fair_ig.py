"""
Fair information-gain feature weighting.

This reuses ``information_gain`` from ``FairTree/split_criterions.py``
to derive a static prior over features for the evolutionary operators. 
For each usable feature we take the best-over-thresholds value of

    fair_ig(feature) = IG(target) - IG(sensitive)

on the full training set (i.e. root-level splits). It clips values to non-negative, and normalise to a
probability vector. Sampling and mutation can then bias feature selection toward attributes
that separate the target without leaking the sensitive attribute """
import sys
import os

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'FairTree'))
from split_criterions import information_gain


def fair_ig_feature_weights(X, y_int, s, meta, usable):
    """
    Compute a normalised fair-information-gain weight per usable feature.

    :param X: training design matrix (n, d), numeric
    :param y_int: integer target labels (n,)
    :param s: sensitive attribute values (n,)
    :param meta: per-column threshold metadata (see genome.build_feature_meta)
    :param usable: list of feature indices with candidate thresholds
    :return: probability weights aligned with ``usable`` (sums to 1)
    """
    y_list = [[v] for v in y_int.tolist()]  # rows with the label at index -1
    s_list = list(s)
    target_classes = sorted(set(y_int.tolist()))
    sensitive_classes = list(set(s_list))

    weights = np.zeros(len(usable), dtype=np.float64)
    for k, j in enumerate(usable):
        col = X[:, j]
        best = 0.0
        for thr in meta[j]["thresholds"]:
            mask = col <= thr
            if mask.all() or (~mask).all():
                continue
            ds_l = [y_list[i] for i in np.nonzero(mask)[0]]
            ds_r = [y_list[i] for i in np.nonzero(~mask)[0]]
            s_l = [s_list[i] for i in np.nonzero(mask)[0]]
            s_r = [s_list[i] for i in np.nonzero(~mask)[0]]
            gain_y = information_gain(y_list, [ds_l, ds_r], target_classes, -1,
                                      [s_l, s_r], s_list)
            gain_s = information_gain(y_list, [ds_l, ds_r], sensitive_classes, "sensitive",
                                      [s_l, s_r], s_list)
            best = max(best, gain_y - gain_s)
        weights[k] = max(best, 0.0)

    if weights.sum() <= 0:
        return np.full(len(usable), 1.0 / len(usable))
    return weights / weights.sum()

"""
Bi-objective problem for evolutionary decision-tree induction.

Objectives (both minimised), matching the existing experiments exactly:
    f1 = 1 - AUROC   (performance)
    f2 = |SPD|       (unfairness: statistical parity difference, or demographic parity
                      difference in the intersectional case)

Fitness protocol mirrors the hyperparameter EA (``pymoo_algos_one_tree.py``): the tree is
scored by K-fold cross-validation within the training set. Because the genome encodes
structure only, each fold refits the leaf labels on the fold's training portion and scores
on the held-out fold.
"""
import sys
import os

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from pymoo.core.problem import ElementwiseProblem

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'algos_two_trees'))
from utils import statistical_parity_diff
from fairlearn.metrics import (demographic_parity_difference,
                               equalized_odds_difference, MetricFrame)

import predict as pr
import genome as g

# worst-case objective values used when a tree/fold is degenerate
WORST_AUROC_OBJ = 1.0   # 1 - 0.0
WORST_SPD = 1.0


def compute_fairness(preds, y_true, s_vals, unprivileged_group, pos_outcome,
                     fairness_metric="spd", intersectional=False):
    """
    Unfairness of ``preds`` under the chosen metric (lower = fairer). Shared by the
    optimization objective (``GPTreeProblem._fairness``) and the final train/test
    evaluation in ``run_gp.py`` so both use identical definitions.
    """
    s_flat = np.asarray(s_vals).flatten()
    if intersectional:
        return demographic_parity_difference(y_true, preds, sensitive_features=s_flat)
    if fairness_metric == "spd":
        return statistical_parity_diff(preds, s_vals, unprivileged_group, pos_outcome)
    if fairness_metric == "equalized_odds":
        return equalized_odds_difference(y_true, preds, sensitive_features=s_flat)
    if fairness_metric == "accuracy_diff":
        mf = MetricFrame(metrics=accuracy_score, y_true=y_true, y_pred=preds,
                         sensitive_features=s_flat)
        return mf.difference()
    raise ValueError('fairness_metric has to be in ["spd", "equalized_odds", "accuracy_diff"], '
                     'got %r' % fairness_metric)


def _safe_auroc(y_true, preds):
    try:
        return roc_auc_score(y_true, preds)
    except ValueError:
        # only one class present in y_true on this fold
        return 0.5


class GPTreeProblem(ElementwiseProblem):
    """Evolve decision-tree topology to trade off AUROC against statistical parity."""

    def __init__(self, folds, X, y_int, s, num_classes, unprivileged_group, pos_outcome,
                 intersectional=False, parsimony=0.0, fairness_metric="spd"):
        """
        :param folds: list of ``(fit_idx, val_idx)`` numpy index arrays over the training set
        :param X: training design matrix (n, d), numeric float
        :param y_int: integer class labels (n,)
        :param s: sensitive attribute values (n,), aligned with X rows
        :param num_classes: number of target classes
        :param parsimony: optional coefficient adding ``parsimony * n_leaves`` to the
            performance objective, as a soft bloat penalty. Default 0.0 leaves the objective
            identical to the existing experiments (kept off for strict comparability).
        :param fairness_metric: second objective, one of "spd", "equalized_odds",
            "accuracy_diff" (see ``compute_fairness``).
        """
        super().__init__(n_var=1, n_obj=2)
        self.folds = folds
        self.X = X
        self.y_int = y_int
        self.s = s
        self.num_classes = num_classes
        self.unprivileged_group = unprivileged_group
        self.pos_outcome = pos_outcome
        self.intersectional = intersectional
        self.parsimony = parsimony
        self.fairness_metric = fairness_metric
        self.global_major = int(np.bincount(y_int, minlength=num_classes).argmax())

    def _fairness(self, preds, y_true, s_vals):
        # Guard degenerate trees/folds: equalized_odds and accuracy_diff can raise or
        # return nan when a group or class is missing on a fold. Treat as worst-case.
        try:
            val = compute_fairness(preds, y_true, s_vals, self.unprivileged_group,
                                   self.pos_outcome, self.fairness_metric, self.intersectional)
        except Exception:
            return WORST_SPD
        if val is None or not np.isfinite(val):
            return WORST_SPD
        return float(val)

    def _evaluate(self, x, out, *args, **kwargs):
        tree = x[0]
        aurocs, spds = [], []
        for fit_idx, val_idx in self.folds:
            pr.fit_leaves(tree, self.X, self.y_int, self.num_classes,
                          idx=fit_idx, fallback=self.global_major)
            preds = pr.predict(tree, self.X[val_idx])
            y_val = self.y_int[val_idx]
            s_val = self.s[val_idx]
            aurocs.append(_safe_auroc(y_val, preds))
            spds.append(self._fairness(preds, y_val, s_val))

        perf = 1.0 - float(np.mean(aurocs))
        if self.parsimony:
            perf += self.parsimony * g.num_leaves(tree)
        out["F"] = [perf, float(np.mean(spds))]

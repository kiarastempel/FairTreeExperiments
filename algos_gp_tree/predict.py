"""
Leaf fitting and vectorized prediction for evolved decision trees.

The genome (see ``genome.py``) encodes split structure only; leaf class labels are fit here
because the majority training class reaching each leaf. Both routines walk the tree over numpy
index masks, so they are vectorized over the whole dataset (no per-row Python recursion).
"""
import numpy as np


def _majority(y_int, num_classes):
    """Majority class of an integer label vector; -1 if empty."""
    if y_int.size == 0:
        return -1
    return int(np.bincount(y_int, minlength=num_classes).argmax())


def fit_leaves(node, X, y_int, num_classes, idx=None, fallback=0):
    """
    Assign each leaf the majority class of the training samples routed to it (in place).

    Empty leaves inherit the majority class of the nearest non-empty ancestor (passed down
    as ``fallback``), and ultimately the global majority.

    :param node: tree root
    :param X: design matrix (n, d), numeric
    :param y_int: integer class labels (n,), values in ``0..num_classes-1``
    :param num_classes: number of target classes
    :param idx: sample indices reaching this node (defaults to all rows)
    :param fallback: majority class to use if this subtree receives no samples
    """
    if idx is None:
        idx = np.arange(X.shape[0])

    if node.is_leaf:
        node.label = _majority(y_int[idx], num_classes) if idx.size else fallback
        if node.label == -1:
            node.label = fallback
        return

    node_major = _majority(y_int[idx], num_classes) if idx.size else fallback
    if node_major == -1:
        node_major = fallback

    feat, thr = node.label, node.threshold
    mask = X[idx, feat] <= thr
    fit_leaves(node.children[0], X, y_int, num_classes, idx[mask], node_major)
    fit_leaves(node.children[1], X, y_int, num_classes, idx[~mask], node_major)


def predict(node, X):
    """
    Predict integer class labels for every row of ``X`` (must have fitted leaves).

    :param node: tree root with fitted leaf labels
    :param X: design matrix (n, d), numeric
    :return: predicted labels (n,), dtype int64
    """
    out = np.empty(X.shape[0], dtype=np.int64)
    _route(node, X, np.arange(X.shape[0]), out)
    return out


def _route(node, X, idx, out):
    if node.is_leaf:
        out[idx] = node.label
        return
    feat, thr = node.label, node.threshold
    mask = X[idx, feat] <= thr
    _route(node.children[0], X, idx[mask], out)
    _route(node.children[1], X, idx[~mask], out)

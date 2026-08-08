"""
Genome for evolutionary decision-tree induction.

An individual *is* a decision tree. It reuses the existing ``Node`` structure from
``FairTree/fair_decision_tree.py``:

- **internal node:** ``is_leaf=False``, ``label`` = *feature column index* (int),
  ``threshold`` = float, ``children`` = ``[left, right]`` where ``left`` is the ``<=`` branch.
- **leaf:** ``is_leaf=True``, ``label`` = predicted class (filled in later by
  ``predict.fit_leaves``; ``None`` until then), ``threshold=None``, ``children=[]``.

Note the genome encodes *split structure only*. Leaf labels are not part of the genome.
They are fit as the majority training class reaching each leaf (see ``predict.py``). This
keeps the genetic operators simple and makes per-fold cross-validation meaningful (leaves
are refit on each fold's training portion).
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'FairTree'))

from fair_decision_tree import Node  # reuse the existing node structure


def build_feature_meta(X, max_thresholds=32, rng=None, binning="midpoints"):
    """
    Precompute, for each feature column, the candidate thresholds an internal node may use.

    - binary (0/1) columns  -> the single meaningful split point ``[0.5]``
    - continuous columns     -> candidate thresholds according to ``binning`` (see below)
    - constant columns       -> no candidates (feature is never selected for a split)

    Two binning schemes for continuous columns:

    - ``"midpoints"`` (default): midpoints between consecutive unique values, i.e. candidate
      splits sit right where the data changes. Potentially many candidates, so the list is
      randomly subsampled down to ``max_thresholds`` when it exceeds that cap. Here
      ``max_thresholds`` acts as an *upper bound*.
    - ``"uniform"``: ``max_thresholds`` evenly spaced thresholds across the observed value
      range ``[min, max]`` of the column (interior points of a linspace). This mirrors a
      fixed-bin scheme like the greedy learner's ``threshold_binning`` and makes
      ``max_thresholds`` an *exact* count (modulo duplicate removal on very narrow ranges).

    Note the two schemes differ in candidate *placement*, not just count: "midpoints" hugs
    the data, "uniform" spreads evenly regardless of where points lie. On skewed features
    they select different split points, so ``max_thresholds`` alone cannot turn one into the
    other. Also, thresholds are computed once on the full training matrix passed in (not
    per node), independent of the chosen scheme.

    :param X: training design matrix, shape (n, d), numeric
    :type X: np.ndarray
    :param max_thresholds: upper bound ("midpoints") or exact count ("uniform") of candidate
        thresholds per continuous feature
    :param binning: ``"midpoints"`` or ``"uniform"``
    :return: list of ``dict`` (one per column) with keys ``kind`` and ``thresholds`` (np.ndarray),
             and the list of usable feature indices (those with >=1 candidate threshold)
    :rtype: tuple[list[dict], list[int]]
    """
    import numpy as np

    if rng is None:
        rng = np.random.default_rng(0)
    if binning not in ("midpoints", "uniform"):
        raise ValueError('binning must be "midpoints" or "uniform", got %r' % binning)

    meta = []
    usable = []
    for j in range(X.shape[1]):
        col = X[:, j]
        uniq = np.unique(col)
        if uniq.size <= 1:
            meta.append({"kind": "constant", "thresholds": np.array([], dtype=np.float64)})
            continue
        if np.all(np.isin(uniq, (0.0, 1.0))):
            meta.append({"kind": "binary", "thresholds": np.array([0.5])})
            usable.append(j)
            continue

        if binning == "uniform":
            # exactly max_thresholds evenly spaced interior points across [min, max];
            # np.unique guards against duplicates when the range is very narrow
            lo, hi = float(col.min()), float(col.max())
            thr = np.linspace(lo, hi, max_thresholds + 2)[1:-1]
            thr = np.unique(thr)
        else:  # "midpoints"
            # midpoints between consecutive unique values, randomly capped at max_thresholds
            thr = (uniq[:-1] + uniq[1:]) / 2.0
            if thr.size > max_thresholds:
                idx = np.sort(rng.choice(thr.size, size=max_thresholds, replace=False))
                thr = thr[idx]

        meta.append({"kind": "continuous", "thresholds": thr.astype(np.float64)})
        usable.append(j)
    return meta, usable


def make_leaf():
    """Create an unlabelled leaf (label filled in by fit_leaves)."""
    return Node(is_leaf=True, label=None, threshold=None)


def make_split(feature_idx, threshold, left, right):
    """Create an internal split node with two children (left = ``<=`` branch)."""
    node = Node(is_leaf=False, label=int(feature_idx), threshold=float(threshold))
    node.children = [left, right]
    return node


def random_threshold(meta, feature_idx, rng):
    """Sample a candidate threshold for ``feature_idx`` from its precomputed set."""
    cands = meta[feature_idx]["thresholds"]
    return float(cands[rng.integers(cands.size)])


def pick_feature(usable, rng, feat_p=None):
    """Choose a feature index from ``usable``, optionally biased by a prior ``feat_p``."""
    if feat_p is None:
        return int(usable[rng.integers(len(usable))])
    return int(usable[rng.choice(len(usable), p=feat_p)])


def random_tree(meta, usable, rng, max_depth, full=False, p_leaf=0.5, feat_p=None):
    """
    Grow a random tree using the classic grow/full schemes (ramped half-and-half is
    obtained by mixing ``full`` across the initial population; see operators.TreeSampling).

    :param meta: per-column threshold metadata from ``build_feature_meta``
    :param usable: list of feature indices that have candidate thresholds
    :param max_depth: maximum tree depth (root is depth 0)
    :param full: if True, always split until ``max_depth`` (full method); else stochastic (grow)
    :param p_leaf: probability of emitting a leaf early in the grow method
    :param feat_p: optional probability weights over ``usable`` (e.g. fair-IG prior)
    """
    def build(depth):
        # forced leaf at max depth, or (grow) a stochastic early stop below the root
        if depth >= max_depth or (not full and depth > 0 and rng.random() < p_leaf):
            return make_leaf()
        feat = pick_feature(usable, rng, feat_p)
        thr = random_threshold(meta, feat, rng)
        return make_split(feat, thr, build(depth + 1), build(depth + 1))

    if not usable:
        return make_leaf()
    return build(0)


# utilities
def copy_tree(node):
    """Deep-copy a tree of ``Node`` objects (structure + split params)."""
    if node.is_leaf:
        return Node(is_leaf=True, label=node.label, threshold=node.threshold)
    new = Node(is_leaf=False, label=node.label, threshold=node.threshold)
    new.children = [copy_tree(c) for c in node.children]
    return new


def depth(node):
    """Depth of the tree (a single leaf has depth 0)."""
    if node.is_leaf:
        return 0
    return 1 + max(depth(c) for c in node.children)


def size(node):
    """Total number of nodes (internal + leaves)."""
    if node.is_leaf:
        return 1
    return 1 + sum(size(c) for c in node.children)


def num_leaves(node):
    if node.is_leaf:
        return 1
    return sum(num_leaves(c) for c in node.children)


def all_nodes(node, include_leaves=True):
    """
    Return a flat list of ``(node, parent, child_index)`` triples for random selection
    by the genetic operators. The root appears as ``(root, None, None)``.
    """
    out = []

    def walk(n, parent, ci):
        if n.is_leaf:
            if include_leaves:
                out.append((n, parent, ci))
        else:
            out.append((n, parent, ci))
            for i, c in enumerate(n.children):
                walk(c, n, i)

    walk(node, None, None)
    return out


def enforce_max_depth(node, max_depth, cur=0):
    """
    Prune the tree in place so its depth does not exceed ``max_depth``: any internal node
    at ``max_depth`` is collapsed into a leaf.
    """
    if node.is_leaf:
        return
    if cur >= max_depth:
        node.is_leaf = True
        node.label = None
        node.threshold = None
        node.children = []
        return
    for c in node.children:
        enforce_max_depth(c, max_depth, cur + 1)


def fingerprint(node):
    """
    Cheap hash function for duplicate elimination. Encodes topology + split
    params (feature, rounded threshold); ignores leaf labels (not part of the genome).
    """
    if node.is_leaf:
        return "L"
    return "(%d<=%.4f %s %s)" % (
        node.label, node.threshold,
        fingerprint(node.children[0]), fingerprint(node.children[1]),
    )

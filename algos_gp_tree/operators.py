"""
pymoo custom operators over object-encoded decision-tree genomes.

The population is an ``object``-dtype array and mating/evolution is defined on these operators. 
"""
import numpy as np
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.duplicate import ElementwiseDuplicateElimination

import genome as g


def _pick_node(nodes, rng, p_internal=0.9):
    """Pick a (node, parent, child_index) triple, preferring internal nodes when present."""
    internal = [t for t in nodes if not t[0].is_leaf]
    if internal and rng.random() < p_internal:
        return internal[rng.integers(len(internal))]
    return nodes[rng.integers(len(nodes))]


class TreeSampling(Sampling):
    """Ramped half-and-half initialisation: mix grow/full over a range of depth caps."""

    def __init__(self, meta, usable, max_depth, rng, feat_p=None):
        super().__init__()
        self.meta = meta
        self.usable = usable
        self.max_depth = max_depth
        self.rng = rng
        self.feat_p = feat_p

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, 1), None, dtype=object)
        for i in range(n_samples):
            full = (i % 2 == 0)
            # ramp the depth cap in [1, max_depth] for population diversity
            d = int(self.rng.integers(1, self.max_depth + 1))
            X[i, 0] = g.random_tree(self.meta, self.usable, self.rng, d, full=full,
                                    feat_p=self.feat_p)
        return X


class TreeCrossover(Crossover):
    """Subtree-swap crossover producing two offspring, with depth repair."""

    def __init__(self, max_depth, rng):
        super().__init__(n_parents=2, n_offsprings=2)
        self.max_depth = max_depth
        self.rng = rng

    def _do(self, problem, X, **kwargs):
        _, n_matings, _ = X.shape
        Y = np.full((self.n_offsprings, n_matings, 1), None, dtype=object)
        for k in range(n_matings):
            a = g.copy_tree(X[0, k, 0])
            b = g.copy_tree(X[1, k, 0])

            node_a, par_a, ci_a = _pick_node(g.all_nodes(a), self.rng)
            node_b, par_b, ci_b = _pick_node(g.all_nodes(b), self.rng)
            sub_a = g.copy_tree(node_a)
            sub_b = g.copy_tree(node_b)

            # child 1 = a with its crossover point replaced by b's subtree
            if par_a is None:
                child1 = sub_b
            else:
                par_a.children[ci_a] = sub_b
                child1 = a
            # child 2 = b with its crossover point replaced by a's subtree
            if par_b is None:
                child2 = sub_a
            else:
                par_b.children[ci_b] = sub_a
                child2 = b

            g.enforce_max_depth(child1, self.max_depth)
            g.enforce_max_depth(child2, self.max_depth)
            Y[0, k, 0] = child1
            Y[1, k, 0] = child2
        return Y


class TreeMutation(Mutation):
    """
    Structural mutation. One operator is chosen per individual among:
    threshold-perturb, feature-swap, grow-leaf, prune-to-leaf. All respect ``max_depth``
    (grow is followed by depth repair, so growth at the depth cap is a no-op).
    """

    def __init__(self, meta, usable, max_depth, rng, prob=0.9, feat_p=None):
        super().__init__()
        self.meta = meta
        self.usable = usable
        self.max_depth = max_depth
        self.rng = rng
        self.prob = prob
        self.feat_p = feat_p

    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            if self.rng.random() > self.prob:
                continue
            X[i, 0] = self._mutate(g.copy_tree(X[i, 0]))
        return X

    def _mutate(self, tree):
        nodes = g.all_nodes(tree)
        internal = [t for t in nodes if not t[0].is_leaf]
        leaves = [t for t in nodes if t[0].is_leaf]

        ops = ["perturb", "feature", "grow", "prune"]
        op = ops[self.rng.integers(len(ops))]

        if op == "perturb" and internal:
            node, _, _ = internal[self.rng.integers(len(internal))]
            node.threshold = g.random_threshold(self.meta, node.label, self.rng)

        elif op == "feature" and internal:
            node, _, _ = internal[self.rng.integers(len(internal))]
            feat = g.pick_feature(self.usable, self.rng, self.feat_p)
            node.label = feat
            node.threshold = g.random_threshold(self.meta, feat, self.rng)

        elif op == "grow" and leaves:
            node, _, _ = leaves[self.rng.integers(len(leaves))]
            feat = g.pick_feature(self.usable, self.rng, self.feat_p)
            node.is_leaf = False
            node.label = feat
            node.threshold = g.random_threshold(self.meta, feat, self.rng)
            node.children = [g.make_leaf(), g.make_leaf()]
            g.enforce_max_depth(tree, self.max_depth)

        elif op == "prune" and internal:
            node, _, _ = internal[self.rng.integers(len(internal))]
            node.is_leaf = True
            node.label = None
            node.threshold = None
            node.children = []

        return tree


class TreeDuplicateElimination(ElementwiseDuplicateElimination):
    """Eliminate structurally identical trees via a cheap fingerprint."""

    def is_equal(self, a, b):
        return g.fingerprint(a.X[0]) == g.fingerprint(b.X[0])

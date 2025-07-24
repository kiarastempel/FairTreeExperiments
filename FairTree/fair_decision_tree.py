#  Copyright (c) 2023 Kiara Stempel
#  All rights reserved.

import time
import numpy as np
import copy
from FairTree.split_criterions import *
from scipy.spatial.distance import pdist, squareform
from collections import Counter
# from utils import plot_split_values_per_feature
import warnings


class FairTree:

    def __init__(self, data, attributes, idx_target, unprivileged_group, pos_outcome,
                 threshold_binning=None, idx_sensitive=None, sensitive=None,
                 leaf_outcome="majority", split_criterion="information_gain", sens_threshold=None, gamma=None,
                 backtracking=False):
        """
        Initializes a fair decision tree

        :param data: data for training the decision tree (holding data instances row for row, columns correspond to attribute)
        :type data: DataFrame
        :param attributes: names of all attributes/input features that are used for training the tree (without the feature corrected in the tree, if stated)
        :type attributes: list[str]
        :param idx_target: index of the temporary target attribute in 'data' (or attribute for the fair correction)
        :type idx_target: int
        :param idx_sensitive: index of the sensitive attribute
        :type idx_sensitive: int
        :param sensitive: values of the sensitive attribute in the same order as in the dataset
        :type sensitive: list[Any]
        :param leaf_outcome: Indicates which method should be used to calculate the outcome in the leaves, has to be in ['mean', 'mean_set_to_range', 'mean_of_range', 'linear_regression']
        :type leaf_outcome: str
        :param split_criterion: Indicates which method is used in split criterion, has to be in ["correlation_coefficient", "mse"]
        :type split_criterion: str
        :param threshold_binning: Test only every nth threshold for splitting numerical attributes in the correcting tree (i.e. set to 1 if every threshold should be tested)
        :type threshold_binning: int
        """
        warnings.filterwarnings("ignore")
        self.tree = None
        self.numAttributes = len(attributes)
        # list of all features, including temp target variable
        self.all_attributes = list(data.columns)
        # only those attributes used as input variables for learning the tree and making predictions later
        self.used_attributes = attributes
        self.attrValues = {}  # saves for each attribute (=key) the list of possible values
        for attribute in self.used_attributes:
            # in case an attribute is categorical, we save all possible values it can take on
            self.attrValues[attribute] = data[attribute].unique()
            # in case it is numeric, we only save that information
            if 'int' in str(type(self.attrValues[attribute][0])) or 'float' in str(type(self.attrValues[attribute][0])):
                self.attrValues[attribute] = ["continuous"]
        self.data = data.to_numpy().tolist()
        self.idx_target = idx_target
        # array with strings of class names (classification) or all possible values (regression)
        self.target_classes = data[self.all_attributes[self.idx_target]].unique()
        self.sensitive_classes = list(set(sensitive))

        # might be needed (depends on implementation)
        self.threshold_binning = threshold_binning
        self.leaf_outcome_method = leaf_outcome
        self.split_criterion = split_criterion

        self.sensitive_data = sensitive
        self.idx_sensitive = idx_sensitive
        self.unprivileged_group = unprivileged_group
        self.pos_outcome = pos_outcome
        self.sens_threshold = sens_threshold
        self.gamma = gamma
        self.backtracking = backtracking

    def print_tree(self):
        """
        Prints the decision tree's string representation.
        """
        if self.tree is None:
            print("Empty tree")
        elif self.tree.is_leaf:
            print("--" + str(self.tree.label))
        else:
            self.print_node(self.tree)

    def print_node(self, node, indent=""):
        """
        Recursive function for printing the decision tree's string representation, starting from the root node.
        :param node: current node of the tree, which is considered as the root of the subtree of which we want to build
        the dot representation (i.e. initially, it is the root node of the complete tree)
        :type node: Node
        :param indent: saves the part of the string representation that is already finished, for appending the following
        part(s)
        :type indent: str
        :return: string representation of current (sub)tree for printing the tree
        :rtype: str
        """
        if not node.is_leaf:
            if type(node.threshold) is list:
                # discrete
                for index, child in enumerate(node.children):
                    if child.is_leaf:
                        print(indent + str(node.label) + " = " + str(node.threshold[index]) + " : " + str(child.label))
                    else:
                        print(indent + str(node.label) + " = " + str(node.threshold[index]) + " : ")
                        self.print_node(child, indent + "|	")
            else:
                # numerical
                leftChild = node.children[0]
                rightChild = node.children[1]
                if leftChild.is_leaf:
                    print(indent + str(node.label) + " <= " + str(node.threshold) + " : " + str(leftChild.label))
                else:
                    print(indent + str(node.label) + " <= " + str(node.threshold) + " : ")
                    self.print_node(leftChild, indent + "|	")

                if rightChild.is_leaf:
                    print(indent + str(node.label) + " > " + str(node.threshold) + " : " + str(rightChild.label))
                else:
                    print(indent + str(node.label) + " > " + str(node.threshold) + " : ")
                    self.print_node(rightChild, indent + "|	")

    def fit(self, min_samples_leave=40, max_depth=30, type="performance", h_limit=0.01):
        """
        Build a fair decision tree from the training set (self.data). It can be a classifier or regressor, depending on
        the target variable. If sens_threshold is set for gain_s and self.tree contains None after fitting, it means
        that no tree can be found that fulfills the threshold condition.

        :param min_samples_leave: minimum number of training instances that should end in a leaf node of the tree
        :type min_samples_leave: int
        :param max_depth: maximum depth that the trained tree should have
        :type max_depth: int
        """
        self.tree = self.recursive_generate_tree(self.data, self.used_attributes, self.sensitive_data,
                                                 min_samples_leave, max_depth, type, h_limit)

    def recursive_generate_tree(self, current_data, current_attributes, current_sens_data,
                                min_samples_leave, max_depth, tree_type, h_limit):
        """
        Recursive function for building a fair decision tree regressor

        :param current_data: data consisting of all instances ending up in this subtree
        :type current_data: list[list[Any]]
        :param current_attributes: attributes that can still be used for splitting
        :type current_attributes: list[str]
        :param current_sens_data: values of sensitive attribute for current data
        :type current_sens_data: list[Any]
        :param min_samples_leave: minimum number of training instances that should end in a leaf node of the tree
        :type min_samples_leave: int
        :param max_depth: maximum depth that the trained tree should have
        :type max_depth: int
        :return: root node of tree
        :rtype: Node
        """
        # print("level with", len(current_data), "instances ( max depth =", max_depth, ")")
        # print("depth", max_depth)
        if len(current_data) == 0:
            # Fail: no data left
            return Node(True, "Fail", None)
        elif len(current_attributes) == 0:
            # there are no attributes left
            # so return a leave node with either mean value as outcome or a linear regression model
            leaf_model = lo.get_leaf_outcome(self.leaf_outcome_method, current_data, -1, list(self.target_classes))
            return Node(True, leaf_model, None)
        else:
            # for backtracking: if the chosen attribute doesn't lead to a feasible solution, remove it from attributes
            # list and try next
            current_attribute_copy = copy.deepcopy(current_attributes)
            while len(current_attribute_copy) > 0:
                # find best split:
                # if in the current node of the tree we have still instances and attributes, then look for the attribute
                # (and in case of numerical attributes in addition a corresponding threshold) that splits the data such
                # that the target attribute is described best
                (best, best_threshold, splits, sens_data_splits) = self.find_best_split(
                    current_data, current_attribute_copy, current_sens_data, min_samples_leave, tree_type, h_limit)
                # check if the tree should be allowed to grow at this node (max depth/min samples reached?)
                if best != -1 and best != -2 and max_depth - 1 > 0:
                    remainingAttributes = current_attributes[:]
                    # only remove attribute from list if it is discrete/categorical
                    if self.is_attr_discrete(best):
                        remainingAttributes.remove(best)
                    # in case we have a categorical attribute, "best_threshold" contains the values which that attribute
                    # can take on
                    node = Node(False, best, best_threshold)
                    # grow tree by recursively calculating the child node(s) of the current node
                    childs = []
                    child_is_none = False
                    for idx, subset in enumerate(splits):
                        child = self.recursive_generate_tree(subset, remainingAttributes,
                                                             sens_data_splits[idx],
                                                             min_samples_leave,
                                                             max_depth - 1,
                                                             tree_type,
                                                             h_limit)
                        if child is None:
                            child_is_none = True
                            break
                        else:
                            childs.append(child)
                    if child_is_none:
                        current_attribute_copy.remove(best)
                        continue
                    node.children = childs
                    return node
                elif best == -2:
                    # gain_s was too high for all possible splits: backtrack to father node
                    return None
                else:
                    # if the pruning criterion is reached (i.e. less than min_samples in a produced leave or max depth)
                    # for each possible attribute, then return a leaf node (with either a mean variant or linear
                    # regression model as outcome)
                    leaf_model = lo.get_leaf_outcome(self.leaf_outcome_method, current_data, -1, list(self.target_classes))
                    return Node(True, leaf_model, None)
            return None

    def find_best_split(self, current_data, current_attributes, current_sens_data, min_samples_leave, tree_type, h_limit):
        """
        Find attribute and the corresponding threshold that provide the best
        split in terms of MSE/correlation/... improvement

        :param current_data: dataset that should be split
        :type current_data: list[list[Any]]
        :param current_attributes: attributes that can still be used for splitting
        :type current_attributes: list[str]
        :param current_sens_data: values of sensitive attribute for current data
        :type current_sens_data: list[Any]
        :param min_samples_leave: minimum number of training instances that should end in a leaf node of the tree
        :type min_samples_leave: int
        :return: best attribute, corresponding best threshold (or None), dataset split into subsets according to the chosen best attribute, sensitive values also split into subsets according to best attribute
        :rtype: tuple[str, float, list[list[list[Any]]], list[list[float]], list[list[float]], list[list[Any]], list[list[Any]]
        """
        splits = []
        sens_data_splits = []

        # variable for saving the best improvement so far compared to the previous split
        max_diff = -1 * float("inf")
        # save index of best attribute
        best_attribute = -2
        # threshold value for continuous attributes, set to None for discrete attributes
        best_threshold = None
        diffs = {}
        for attribute in current_attributes:
            diffs[attribute] = []
            index_of_attribute = self.all_attributes.index(attribute)
            if self.is_attr_discrete(attribute):
                # discrete attribute
                # discrete_split() calculates for the given specific attribute the improvement of e.g. information gain
                # compared to the previous split/node and returns the subsets of the data after splitting
                information_best = self.discrete_split(current_data, attribute, index_of_attribute, min_samples_leave,
                                                       current_sens_data, tree_type, h_limit)
                diffs[attribute].append(information_best[4])
            else:
                # numerical attribute
                # numerical_split() find for one specific attribute the best threshold and returns in that order:
                # best_attribute (which corresponds to the given specific attribute), best_threshold, splits,
                # weights_splits, sens_data_splits, max_diff, diffs
                information_best = self.numerical_split(
                    current_data, attribute, index_of_attribute, min_samples_leave, diffs, current_sens_data,
                    tree_type, h_limit)
            if information_best[4] >= max_diff:
                splits = information_best[2]
                sens_data_splits = information_best[3]
                max_diff = information_best[4]
                best_attribute = information_best[0]
                best_threshold = information_best[1]
        return best_attribute, best_threshold, splits, sens_data_splits

    def is_attr_discrete(self, attribute):
        """
        Check if given attribute is discrete.

        :param attribute: name of attribute to check
        :type attribute: str
        :return: True if attribute is discrete, False if not
        :rtype: bool
        """
        if attribute not in self.used_attributes:
            raise ValueError("Attribute not listed")
        elif len(self.attrValues[attribute]) == 1 and self.attrValues[attribute][0] == "continuous":
            return False
        else:
            return True

    def numerical_split(self, current_data, attribute, index_of_attribute, min_samples_leave, diffs,
                        current_sens_data, tree_type, h_limit):
        """
        Split if attribute is numerical: Try all thresholds for the given attribute for splitting the data and return that attribute/threshold pair
        including all splitting information which causes the maximal correlation or maximum decrease of MSE.

        :param current_data: dataset that should be split
        :type current_data: list[list[Any]]
        :param attribute: attribute that is currently considered for splitting the data
        :type attribute: str
        :param index_of_attribute: index of attribute that is currently considered for splitting the data
        :type index_of_attribute: int
        :param min_samples_leave: minimum number of training instances that should end in a leaf node of the tree
        :type min_samples_leave: int
        :param diffs: for saving the best reached difference in the specified criterion for each attribute
        :type diffs: dict
        :param current_sens_data: values of sensitive attribute for current data
        :type current_sens_data: list[Any]
        :return: best attribute, corresponding best threshold (or None), dataset split into subsets according to the chosen best attribute, sensitive values also split into subsets according to best attribute, maximum reached diff
        :rtype: tuple[str, float, list[list[list[Any]]], list[list[float]], list[list[float]], list[list[Any]], list[list[Any]], float, dict[str, float]
        """
        # sort the data according to the column, then try all possible adjacent pairs and
        # choose the one that yields maximum gain
        sorted_data = sorted(
            zip(current_data, current_sens_data),
            key=lambda x: x[0][index_of_attribute])
        a = sorted_data
        current_data, current_sens_data = zip(*sorted_data)
        # following line is for sorting only data array independently
        # current_data = sorted(current_data, key=lambda x: x[index_of_attribute])
        splits = []
        sens_data_splits = []
        max_diff = -1 * float("inf")
        if self.backtracking:
            best_attribute = -2
        else:
            best_attribute = -1
        best_threshold = None

        threshold_idx = list(range(0, len(current_data) - 1))
        if 'float' in str(type(self.threshold_binning)):
            # test n% of all thresholds, but in equal steps
            n = int(1 / self.threshold_binning)
        elif 'int' in str(type(self.threshold_binning)):
            # "binning": test only n thresholds for splitting
            n = int(len(threshold_idx) / self.threshold_binning)
        else:
            raise ValueError("For threshold_binning, you have to pass an integer or float between 0 and 1.")
        if n == 0:
            threshold_idx = [int(len(threshold_idx) / 2)]
        else:
            threshold_idx = threshold_idx[0::n]
        for index, j in enumerate(threshold_idx):
            threshold_tests = 0
            diff = 0
            if index + 1 < len(threshold_idx):
                next_j = threshold_idx[index + 1]
                if current_data[j][index_of_attribute] != current_data[next_j][index_of_attribute]:
                    threshold_tests += 1
                    # the groups less and greater than the threshold become the two data subsets
                    # (used for gain calculation)
                    threshold = (current_data[j][index_of_attribute] + current_data[j + 1][index_of_attribute]) / 2
                    less = []
                    sensitive_less = []
                    greater = []
                    sensitive_greater = []
                    for idx, row in enumerate(current_data):
                        if row[index_of_attribute] > threshold:
                            greater.append(row)
                            sensitive_greater.append(current_sens_data[idx])
                        else:
                            less.append(row)
                            sensitive_less.append(current_sens_data[idx])

                    if len(less) >= min_samples_leave and len(greater) >= min_samples_leave:
                        h_correct = True
                        if tree_type == "fair":
                            for s in [sensitive_less, sensitive_greater]:
                                # calculate entropy of S
                                e = entropy(s, self.sensitive_classes, "sensitive", s)
                                if e <= h_limit:
                                    h_correct = False
                        elif tree_type == "performance":
                            for s in [less, greater]:
                                # calculate entropy of y
                                e = entropy(s, self.target_classes, self.idx_target, s)
                                if e >= h_limit:
                                    h_correct = False
                        if h_correct:
                            if self.backtracking:
                                gain_s = information_gain(current_data, [less, greater], self.sensitive_classes, "sensitive",
                                                          [sensitive_less, sensitive_greater], current_sens_data)
                                #print(gain_s)
                                if gain_s <= self.sens_threshold:
                                    # calculate gain on y
                                    diff = self.gain(current_data, index_of_attribute, [less, greater],
                                                     [sensitive_less, sensitive_greater], current_sens_data)
                                else:
                                    # attribute can't be chosen due to fairness reasons, 'encode' as -2
                                    diff = -2
                            else:
                                diff = self.gain(current_data, index_of_attribute, [less, greater],
                                                 [sensitive_less, sensitive_greater], current_sens_data)

                            # max(-2, -1) = -1 --> if there is at least one split that is fair enough but has too few
                            # samples in a child node, we create a leave
                            if diff >= max_diff:
                                splits = [less, greater]
                                sens_data_splits = [sensitive_less, sensitive_greater]
                                max_diff = diff
                                best_attribute = attribute
                                best_threshold = threshold
                    else:
                        if best_attribute == -2:
                            best_attribute = -1
            if threshold_tests == 0:
                if best_attribute == -2:
                    best_attribute = -1
            diffs[attribute].append(diff)
        if best_attribute == -2:
            print("hi")
        return best_attribute, best_threshold, splits, sens_data_splits, max_diff

    def discrete_split(self, current_data, attribute, index_of_attribute, min_samples_leave, current_sensitive_data,
                       tree_type, h_limit):
        """
        Split if attribute is discrete: split data into subsets according to the instances' values of that attribute.
        Return subsets and difference of gain

        :param current_data: dataset that should be split
        :type current_data: list[list[Any]]
        :param attribute: attribute that is currently considered for splitting the data
        :type attribute: str
        :param index_of_attribute: index of attribute that is currently considered for splitting the data
        :type index_of_attribute: int
        :param min_samples_leave: minimum number of training instances that should end in a leaf node of the tree
        :type min_samples_leave: int
        :param current_sensitive_data: values of sensitive attribute for current data
        :type current_sensitive_data: list[Any]
        :return: gain, subsets
        :rtype: tuple[float, list[list[Any]]]
        """
        # split curData into n-subsets, where n is the number of
        # different values of attribute i, resulting in the subsets-list
        valuesForAttribute = self.attrValues[attribute]
        subsets = [[] for a in valuesForAttribute]
        sensitive_subsets = [[] for a in valuesForAttribute]
        diff = -1 * float("inf")
        best_attribute = -2
        for idx, row in enumerate(current_data):
            for index in range(len(valuesForAttribute)):
                if row[index_of_attribute] == valuesForAttribute[index]:
                    subsets[index].append(row)
                    sensitive_subsets[index].append(current_sensitive_data[idx])
                    break
        enough_samples = True
        for s in subsets:
            if len(s) < min_samples_leave:
                enough_samples = False
        # another if for minimal/maximal information
        h_correct = True
        if tree_type == "fair":
            for s in sensitive_subsets:
                # calculate entropy of S
                e = entropy(s, self.sensitive_classes, "sensitive", s)
                if e <= h_limit:
                    h_correct = False
        elif tree_type == "performance":
            for s in subsets:
                # calculate entropy of y
                e = entropy(s, self.target_classes, self.idx_target, s)
                if e >= h_limit:
                    h_correct = False
        if enough_samples and h_correct:
            if self.backtracking:
                gain_s = information_gain(current_data, subsets, self.sensitive_classes, "sensitive",
                                          sensitive_subsets, current_sensitive_data)
                if gain_s <= self.sens_threshold:
                    # calculate gain on y
                    diff = self.gain(current_data, index_of_attribute, subsets, sensitive_subsets,
                                     current_sensitive_data)
                    best_attribute = attribute
                else:
                    # attribute can't be chosen due to fairness reasons, 'encode' as -2
                    diff = -2
            else:
                diff = self.gain(current_data, index_of_attribute, subsets, sensitive_subsets, current_sensitive_data)
                best_attribute = attribute
        else:
            if best_attribute == -2:
                best_attribute = -1
        return best_attribute, list(valuesForAttribute), subsets, sensitive_subsets, diff

    def gain(self, current_data, index_of_attribute, subsets, sensitive_data, current_sens_data):
        """
        Calculate difference (or gain) in value of split criterion according to given split data (i.e., the subsets),
        e.g. MSE/correlation/information gain.

        :param current_sens_data: current subset of the sensitive data, i.e. the subset that corresponds to the current node
        :type current_sens_data: list[Any]
        :param current_data: dataset that should be split
        :type current_data: list[list[Any]]
        :param index_of_attribute: index of attribute that is currently considered for splitting the data
        :type index_of_attribute: int
        :param subsets: splitted data
        :type subsets: list[list[Any]]
        :param sensitive_data: values of sensitive attribute for current data, splitted according to the attribute/threshold we consider at that point
        :type sensitive_data: list[list[Any]]
        :return: difference in specified criterion
        :rtype: float
        """
        # Standard: choose the attribute with maximum correlation
        correlation = mse_difference(current_data, subsets, index_of_attribute)
        return correlation

    def all_same_class(self, data):
        """
        Check if all instances in the given dataset have the same class label

        :param data: dataset
        :type data: list[list[Any]]
        :return: If all instances have the same class label, return this class label, if not, return False.
        :rtype: Any
        """
        for row in data:
            if row[-1] != data[0][-1]:
                return False
        return data[0][-1]

    def predict(self, data):
        if self.tree is None:
            print("Empy tree cannot predict")
            return []
        predictions = []
        for idx, instance in data.iterrows():
            predictions.append(self.predict_instance(instance, self.tree))
        return predictions

    def predict_instance(self, instance, tree):
        if tree.is_leaf:
            prediction = None
            if self.leaf_outcome_method == "mean" or self.leaf_outcome_method == "majority" or self.leaf_outcome_method == "probability":
                prediction = tree.label
            else:
                print("Method of leaf outcome not supported")
            return prediction
        else:
            if type(tree.threshold) is not list:
                # consider only numerical attributes
                if instance[self.used_attributes.index(tree.label)] <= tree.threshold:
                    return self.predict_instance(instance, tree.children[0])
                else:
                    return self.predict_instance(instance, tree.children[1])
            else:
                # then attribute is categorical
                inst_attribute_value = instance[self.all_attributes.index(tree.label)]
                all_attribute_values = tree.threshold
                # look for index of instance' current attribute value in list
                child_idx = all_attribute_values.index(inst_attribute_value)
                return self.predict_instance(instance, tree.children[child_idx])


class Node:

    def __init__(self, is_leaf, label, threshold):
        """
        Initializes a node of the tree.

        :param is_leaf: True if node is a leaf, False if not
        :type is_leaf: bool
        :param label: contains the class label or predicted value for leaves, and contains the name of the attribute
        that was chosen for splitting at that node for inner nodes
        :type label: Any
        :param threshold: contains for numerical attributes the threshold that was chosen for splitting (threshold=None for leaves)
        :type threshold: float
        """
        self.label = label              # splitting attribute / output class
        self.threshold = threshold      # numerical: attribute < threshold, categorical: list of categories
        self.is_leaf = is_leaf          # Bool
        self.children = []              # numerical: 2 children nodes, categorical: #categories children nodes

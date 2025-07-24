import FairTree.leaf_outcomes as lo
import math
import numpy as np
from algos_two_trees.utils import statistical_parity_diff


def information_gain(dataset, subsets, classes, attribute_idx=-1, sensitive_data=None,
                     current_sens_data=None):
    """
    Calculate information gain for split (i.e. the complete dataset split into subsets)

    :param current_sens_data: current subset of the sensitive data, i.e. the subset that corresponds to the current node
    :type current_sens_data: list[Any]
    :param sensitive_data: current sensitive data, splitted according to the attribute/threshold we consider at that point
    :type sensitive_data: list[list[Any]]
    :param attribute_idx: index of attribute of which we calculate the information gain, either the index or "sensitive"
    :type attribute_idx: int or str
    :param classes: list of unique classes of the attribute of which we calculate the gain
    :type classes: list[Any]
    :param dataset: complete dataset, equals the unioned subsets
    :type dataset: list[list[Any]]
    :param subsets: list of subsets of dataset
    :type subsets: list[list[list[Any]]]
    :return: information gain
    :rtype: float
    """
    # input : data and disjoint subsets of it
    # output : information gain
    S = len(dataset)
    # calculate impurity before split
    impurityBeforeSplit = entropy(dataset, classes, attribute_idx, current_sens_data)
    # calculate impurity after split
    weights = [len(subset) / S for subset in subsets]
    impurityAfterSplit = 0
    for i in range(len(subsets)):
        impurityAfterSplit += weights[i] * entropy(subsets[i], classes, attribute_idx, sensitive_data[i])
    # calculate total gain
    totalGain = impurityBeforeSplit - impurityAfterSplit
    return totalGain


def entropy(dataset, classes, attribute_idx, sensitive_data):
    """
    Calculate entropy of given dataset

    :param sensitive_data: subset of the sensitive data of which we want to calculate the "purity" of the classes
    :type sensitive_data: list[Any]
    :param attribute_idx: index of attribute of which we calculate the information gain, either the index or "sensitive"
    :type attribute_idx: int or str
    :param classes: list of unique classes of the attribute of which we calculate the gain
    :type classes: list[Any]
    :param dataset: dataset
    :type dataset: list[list[Any]]
    :return: entropy
    :rtype: float
    """
    S = len(dataset)
    # get unique values of attribute

    if S == 0:
        return 0
    num_classes = [0 for i in classes]
    if attribute_idx == "sensitive":
        for row in sensitive_data:
            classIndex = list(classes).index(row)
            num_classes[classIndex] += 1
    else:
        for row in dataset:
            classIndex = list(classes).index(row[attribute_idx])
            num_classes[classIndex] += 1
    num_classes = [x / S for x in num_classes]
    ent = 0
    for num in num_classes:
        if num != 0:
            ent += num * log(num)
    return ent * -1


def log(x):
    if x == 0:
        return 0
    else:
        return math.log(x, 2)
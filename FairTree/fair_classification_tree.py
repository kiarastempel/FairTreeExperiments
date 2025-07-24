#  Copyright (c) 2023 Kiara Stempel
#  All rights reserved.

import numpy as np
import pandas as pd
import math
from FairTree.fair_decision_tree import FairTree
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from FairTree.split_criterions import *


class FairClassificationTree(FairTree):

    def __init__(self, data, attributes, idx_target, unprivileged_group, pos_outcome,
                 threshold_binning=None, idx_sensitive=None, sensitive=None,
                 leaf_outcome="linear_regression", split_criterion="correlation_coefficient", sens_threshold=None,
                 gamma=None, backtracking=False):
        super().__init__(data, attributes, idx_target, unprivileged_group, pos_outcome,  threshold_binning,
                         idx_sensitive, sensitive, leaf_outcome, split_criterion, sens_threshold, gamma, backtracking)

    def gain(self, current_data, index_of_attribute, subsets, sensitive_data, current_sens_data):
        """
        Calculate difference (or gain) in value of split criterion according to given split data (i.e., the subsets),
        here information gain.

        :param current_sens_data: current subset of the sensitive data, i.e. the subset that corresponds to the current node
        :type current_sens_data: list[Any]
        :param current_data: dataset that should be split
        :type current_data: list[list[Any]]
        :param index_of_attribute: index of attribute that is currently considered for splitting the data
        :type index_of_attribute: int
        :param subsets: splitted data
        :type subsets: list[list[Any]]
        :param sensitive_data: current sensitive data, splitted according to the attribute/threshold we consider at that point
        :type sensitive_data: list[list[Any]]
        :return: difference in specified criterion
        :rtype: float
        """
        if self.split_criterion == "information_gain":
            # choose the attribute with minimum information gain regarding target/sensitive attribute
            diff = information_gain(current_data, subsets, self.target_classes, -1, sensitive_data, current_sens_data)
        elif self.split_criterion == "threshold_constraint":
            # choose the attribute with minimum information gain regarding target/sensitive attribute
            diff = information_gain(current_data, subsets, self.target_classes, -1, sensitive_data, current_sens_data)
            diff_sens = information_gain(current_data, subsets, self.sensitive_classes, "sensitive", sensitive_data,
                                         current_sens_data)
            if self.sens_threshold is not None:
                if diff_sens > self.sens_threshold:  # e.g. 0.001
                    # then the current attribute is not an option
                    # print("sensitive gain", diff_sens)
                    # print("treshold", self.sens_threshold)
                    diff = -1 * float("inf")
        elif self.split_criterion == "gain_s":
            diff_sens = information_gain(current_data, subsets, self.sensitive_classes, "sensitive", sensitive_data,
                                         current_sens_data)
            diff = - 1 * diff_sens
        elif self.split_criterion == "weighted_combi":
            diff_target = information_gain(current_data, subsets, self.target_classes, -1, sensitive_data, current_sens_data)
            diff_sens = information_gain(current_data, subsets, self.sensitive_classes, "sensitive", sensitive_data,
                                         current_sens_data)
            diff = (1 - self.gamma) * diff_target - self.gamma * diff_sens
        elif self.split_criterion == "spd":
            # minimize statistical parity difference
            diff = -1 * spd(current_data, subsets, sensitive_data, current_sens_data,
                            self.unprivileged_group, self.pos_outcome)
        else:
            raise ValueError("split criterion not supported")
        return diff

    def evaluate(self, predictions, labels):
        num_correct_preds = 0
        for i in range(len(predictions)):
            if predictions[i] == labels[i]:
                num_correct_preds += 1
        accuracy = num_correct_preds / len(predictions)
        return accuracy



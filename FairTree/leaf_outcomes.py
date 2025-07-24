#  Copyright (c) 2023 Kiara Stempel
#  All rights reserved.

from sklearn.linear_model import LinearRegression
import numpy as np


def get_major_class(dataset):
    """
    Calculate majority class of all instances in the given dataset

    :param dataset: dataset
    :type dataset: list[list[Any]]
    :parm classes: list of all occurring classes
    :type classes: list[Any]
    :return: majority class
    :rtype: Any
    """
    # extract all unique classes
    classes = list({row[-1] for row in dataset})
    freq = [0] * len(classes)
    for row in dataset:
        index = classes.index(row[-1])
        freq[index] += 1
    maxInd = freq.index(max(freq))
    return classes[maxInd]


def get_probs(dataset, classes):
    """
    Calculate majority class of all instances in the given dataset

    :param dataset: dataset
    :type dataset: list[list[Any]]
    :parm classes: list of all occurring classes
    :type classes: list[Any]
    :return: majority class
    :rtype: Any
    """
    # classes = [0.0, 1.0]  # list({row[-1] for row in dataset})
    freq = [0] * len(classes)
    for row in dataset:
        index = classes.index(row[-1])
        freq[index] += 1

    total = sum(freq)  # Total number of rows in the dataset
    probabilities = [f / total for f in freq]
    return probabilities


def get_mean_label(dataset, attribute, weights=None):
    """
    Get mean value of the given attribute of all instances in the given dataset

    :param dataset: dataset
    :type dataset: list[list[Any]]
    :param attribute: attribute of which the mean value should be calculated (name or index of attribute)
    :type attribute: Any
    :param weights: optional weight for each instance, if None all weights are set to 1.0
    :type weights: list[float]
    :return: mean value
    :rtype: float
    """
    if weights is None:
        weights = [1.0] * len(dataset)
    sum = 0
    for i, row in enumerate(dataset):
        sum += row[attribute] * weights[i]
    return sum / np.sum(weights)


def mse(prediction, dataset, attribute):
    """
    Calculate mean squared error (MSE) of attribute values and predictions of attribute, where the predicted value
    is the mean value of the attribute values of all instances in the dataset

    :param prediction: predicted value for the instances
    :type prediction: float
    :param dataset: dataset for calculating MSE of predictions
    :type dataset: list[list[Any]]
    :param attribute: name or index of attribute for calculating MSE
    :type attribute: Any
    :return: MSE
    :rtype: float
    """
    mse = 0
    for row in dataset:
        error = row[attribute] - prediction
        mse += error * error
    mse /= len(dataset)
    return mse


def mse_of_regression(dataset, attribute):
    # use all attribute besides the one to be corrected of the current tree
    attributes = list(range(len(dataset[0])))
    used_attributes = attributes[:attribute] + attributes[attribute + 1:]
    X = []
    y = []
    for row in dataset:
        X.append(row[used_attributes])
        y.append(row[attribute])
    reg = LinearRegression().fit(X, y)

    mse = 0
    for row in dataset:
        error = row[attribute] - reg.predict([row[attributes[:attribute] + attributes[attribute + 1:]]])
        mse += error * error
    mse /= len(dataset)
    return mse


def leaf_outcome_error(method, dataset, attribute):
    if method == "mean":
        mean = get_mean_label(dataset, attribute)
        return mse(mean, dataset, attribute)
    else:
        raise ValueError("Method for leaf outcome not defined, as to be in {mean}")


def get_leaf_outcome(leaf_outcome, current_data, attribute, classes):
    if leaf_outcome == "mean":
        return get_mean_label(current_data, attribute)
    elif leaf_outcome == "majority":
        return get_major_class(current_data)
    elif leaf_outcome == "probability":
        return get_probs(current_data, classes)
    else:
        raise ValueError("Method for leaf outcome not defined, has to be in {mean, majority}")


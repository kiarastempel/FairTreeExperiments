from fairlearn.metrics import MetricFrame, demographic_parity_difference, selection_rate,\
                              equalized_odds_difference
from sklearn.metrics import accuracy_score
import pandas as pd

def statistical_parity_diff(y_preds, s, unprivileged_group, pos_outcome):
    num_pos_priv = 0
    num_inst_priv = 0
    num_pos_unpriv = 0
    num_inst_unpriv = 0

    for i in range(len(y_preds)):
        group = s[i]
        if unprivileged_group == group:
            num_inst_unpriv += 1
            if y_preds[i] == pos_outcome:
                num_pos_unpriv += 1
        else:
            num_inst_priv += 1
            if y_preds[i] == pos_outcome:
                num_pos_priv += 1
    if num_inst_unpriv == 0 or num_inst_priv == 0:
        return float('inf')
    return abs((num_pos_unpriv / num_inst_unpriv) - (num_pos_priv / num_inst_priv))


def demographic_parity_worst(y_true, y_pred, sensitive_features):
    return demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)


def demographic_parity_average(y_true, y_pred, sensitive_features, debug=False):
    mf = MetricFrame(metrics={"selection_rate": selection_rate}, y_true=y_true,
                     y_pred=y_pred, sensitive_features=sensitive_features)
    if debug:
        print(mf.by_group)
    overall_rate = mf.overall
    avg_dp_diff = (mf.by_group - overall_rate).abs().mean()
    return avg_dp_diff


def equalized_odds_worst(y_true, y_pred, sensitive_features):
    return equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)


def equalized_odds_average(y_true, y_pred, sensitive_features, debug=False):
    mf = MetricFrame(metrics={"accuracy": accuracy_score}, y_true=y_true,
                     y_pred=y_pred, sensitive_features=sensitive_features)
    if debug:
        print(mf.by_group)
    overall_acc = mf.overall
    avg_eo_diff = (mf.by_group - overall_acc).abs().mean()
    return avg_eo_diff


if __name__ == '__main__':
    y_true = [1, 1, 0, 0]
    y_pred = [1, 1, 1, 0]
    sens = pd.DataFrame(['0_0', '1_0', '0_1', '1_1'], columns=['inter'])
    dpa = demographic_parity_average(y_true, y_pred, sensitive_features=sens, debug=True)
    print('demo parity average: {}'.format(dpa))
    dpw = demographic_parity_worst(y_true, y_pred, sensitive_features=sens)
    print('demo parity worst: {}'.format(dpw))
    eoa = equalized_odds_average(y_true, y_pred, sensitive_features=sens, debug=True)
    print('eodds average: {}'.format(eoa))


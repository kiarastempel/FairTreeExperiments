

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

import pandas as pd
import os
from constants import PROJECT_ROOT


def get_compas(file=os.path.join(PROJECT_ROOT, 'data', 'compas-preprocessed.csv')):
    # get Compas data
    df = pd.read_csv(file)
    # One-hot encode categorical features
    categorical_features = ['age_cat', 'race', 'sex', 'c_charge_degree', 'score_text']
    numerical_features = ["priors_count", "decile_score"]
    df_encoded = pd.get_dummies(df, columns=categorical_features)
    # Define features (X), target variable (y), and sensitive attribute (s)
    X = df_encoded.drop(['two_year_recid', 'race_African-American', 'race_Caucasian', 'decile_score'], axis=1)
    y_orig = df_encoded['two_year_recid']  # Target variable
    s_orig = df_encoded['race_African-American']  # Sensitive attribute

    unprivileged_group = 1
    pos_outcome = 0  # 1  # person not rearrested within two years
    return X, y_orig, s_orig, unprivileged_group, pos_outcome


def get_adult(url=os.path.join(PROJECT_ROOT, 'data', 'adult.csv')):
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                    'income']
    df = pd.read_csv(url, header=None, names=column_names, na_values=' ?')
    # Drop rows with missing values
    df = df.dropna().reset_index(drop=True)
    s_orig = df['sex']
    numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    # One-hot encode categorical features
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                            'native-country']
    df_encoded = pd.get_dummies(df, columns=categorical_features)

    # Separate features and target variable
    X = df_encoded.drop(['sex', 'income'], axis=1)
    y_orig = df_encoded['income']

    # encode s
    s = []
    for j in s_orig:
        if "Female" in j:
            s.append(0)
        else:
            s.append(1)

    y = []
    for k in y_orig:
        if '<=50K' in k:
            y.append(0)
        else:
            y.append(1)

    unprivileged_group = 0
    pos_outcome = 1  # " >50K"

    return X, pd.Series(y), s, unprivileged_group, pos_outcome


def get_law(url=os.path.join(PROJECT_ROOT, 'data', 'law_dataset.csv')):
    column_names = ['decile1b', 'decile3', 'lsat', 'ugpa', 'zfygpa', 'zgpa', 'fulltime', 'fam_inc', 'male', 'race',
                    'tier', 'pass_bar']
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', url), delimiter=',')
    # Drop rows with missing values
    df = df.dropna().reset_index(drop=True)
    s_orig = df['race']
    numerical_features = ['decile1b', 'decile3', 'lsat', 'ugpa', 'zfygpa', 'zgpa']

    # One-hot encode categorical features
    categorical_features = ['fulltime', 'fam_inc', 'male', 'tier']
    df_encoded = pd.get_dummies(df, columns=categorical_features)

    # Separate features and target variable
    X = df_encoded.drop(['race', 'pass_bar'], axis=1)
    y_orig = df_encoded['pass_bar']

    unprivileged_group = 0
    pos_outcome = 1  # " >50K"

    return X, pd.Series(y_orig), s_orig, unprivileged_group, pos_outcome


def get_dutch_census(url=os.path.join(PROJECT_ROOT, 'data', 'dutch_census_2001.csv')):
    column_names = ['sex', 'age', 'household_position', 'household_size', 'prev_residence_place', 'citizenship',
                    'country_birth', 'edu_level', 'economic_status', 'cur_eco_activity', 'Marital_status', 'occupation']
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', url), delimiter=',')
    # Drop rows with missing values
    df = df.dropna().reset_index(drop=True)
    s_orig = df['sex']
    numerical_features = []
    # One-hot encode categorical features
    categorical_features = ['age', 'household_position', 'household_size', 'prev_residence_place', 'citizenship',
                    'country_birth', 'edu_level', 'economic_status', 'cur_eco_activity', 'Marital_status']
    df_encoded = pd.get_dummies(df, columns=categorical_features)

    # Separate features and target variable
    X = df_encoded.drop(['sex', 'occupation'], axis=1)
    y_orig = df_encoded['occupation']

    s = []
    for j in s_orig:
        if j == 1:
            s.append(0)
        else:
            s.append(1)

    # Encode y as 0/1
    y = []
    for target in y_orig:
        if '2_1' in target:
            y.append(1)  # high-level occupation (1: legislators, senior officials, managers; 2: professionals)
        else:
            y.append(0)  # '5_4_9' is low-level occupation (5: service workers, 4: clerks, 9: elementary occupations)

    unprivileged_group = 0
    pos_outcome = 1

    return X, pd.Series(y), s, unprivileged_group, pos_outcome


def get_banks():
    column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                    'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'bank.csv'), delimiter=';')

    s_orig = df['age']

    numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

    # One-hot encode categorical features
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    df_encoded = pd.get_dummies(df, columns=categorical_features)

    # Separate features and target variable
    X = df_encoded.drop(['age', 'y'], axis=1)
    y_orig = df_encoded['y']

    # Encode s as based on age threshold (e.g., age < 25 is considered young, otherwise old)
    s = []
    for age in s_orig:
        if int(age) < 25:
            s.append(0)  # Young
        else:
            s.append(1)  # Old

    # Encode y as 0/1
    y = []
    for target in y_orig:
        if 'no' in target:
            y.append(0)
        else:
            y.append(1)

    unprivileged_group = 0
    pos_outcome = 1  # "yes"

    return X, pd.Series(y), s, unprivileged_group, pos_outcome


def get_german(file=os.path.join(PROJECT_ROOT, 'data', 'german-credit.csv')):
    df = pd.read_csv(file, sep=',', header=0)

    integer_features = ["duration", "amount", "installment_rate", "present_residence", "age", "number_credits",
                        "people_liable"]
    binary_features = ["telephone", "foreign_worker"]
    categorical_features = ["status", "credit_history", "purpose", "savings", "employment_duration",
                            "personal_status_sex", "other_debtors", "property", "other_installment_plans", "housing",
                            "job"]

    # Convert numerical features to floats or ints
    for feature in integer_features:
        df[feature] = df[feature].astype('int')

    for feature in binary_features:
        df[feature] = df[feature].map({'yes': 1, 'no': 0})

    s_orig = df['age']

    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=categorical_features)

    # Separate features and target variable
    X = df_encoded.drop(['age', 'credit_risk'], axis=1)
    y_orig = df_encoded['credit_risk']

    # Encode 's' (sensitive attribute) as 0 (young, age <= 25) or 1 (old, age > 25)
    s = [0 if age <= 25 else 1 for age in s_orig]

    # Encode 'y' (target variable) as 0 (bad credit risk) or 1 (good credit risk)
    y = [0 if val == 0 else 1 for val in y_orig]

    # Define unprivileged group and positive outcome
    unprivileged_group = 0  # Young means unprivileged
    pos_outcome = 1  # Good credit risk

    return X, pd.Series(y), s, unprivileged_group, pos_outcome

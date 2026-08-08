import sys
sys.path.append('../')
import pandas as pd
import os
import numpy as np
from constants import PROJECT_ROOT


def data_loader_router(identifier, intersectional):
    if identifier == "Compas":
        X, y, s, unprivileged_group, pos_outcome = get_compas(intersectional=intersectional)
    elif identifier == "Adult":
        X, y, s, unprivileged_group, pos_outcome = get_adult(intersectional=intersectional)
    elif identifier == "Banks":
        X, y, s, unprivileged_group, pos_outcome = get_banks(intersectional=intersectional)
    elif identifier == "German":
        X, y, s, unprivileged_group, pos_outcome = get_german(intersectional=intersectional)
    elif identifier == "Law":
        X, y, s, unprivileged_group, pos_outcome = get_law(intersectional=intersectional)
    elif identifier == "Dutch":
        X, y, s, unprivileged_group, pos_outcome = get_dutch_census(intersectional=intersectional)
    elif identifier == "Folktables_AK":
        X, y, s, unprivileged_group, pos_outcome = get_folktables("AK", intersectional=intersectional)
    elif identifier == "Folktables_HI":
        X, y, s, unprivileged_group, pos_outcome = get_folktables("HI", intersectional=intersectional)
    else:
        raise ValueError("unknown dataset")
    return X, y, s, unprivileged_group, pos_outcome


def get_compas(file=os.path.join(PROJECT_ROOT, 'data', 'compas-preprocessed.csv'), intersectional=False):
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
    if intersectional:
        s_additional = X['sex_Female']
        s_conv = s_additional.astype(int).astype(str)
        s_orig = s_orig.astype(int).astype(str)
        s_orig = s_conv + '_' + s_orig
        s_orig = pd.DataFrame(s_orig, columns=['Female_AfricanAmerican'])
        unprivileged_group = '1_1'
        X = X.drop(['sex_Male', 'sex_Female'], axis=1)
        pos_outcome = 0
    else:
        unprivileged_group = 1
        pos_outcome = 0  # 1  # person not rearrested within two years
    return X, y_orig, s_orig, unprivileged_group, pos_outcome


def get_adult(url=os.path.join(PROJECT_ROOT, 'data', 'adult.csv'), intersectional=False):
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
    if intersectional:
        # we save this for later
        categorical_features.remove('race')

    df_encoded = pd.get_dummies(df, columns=categorical_features)

    # Separate features and target variable
    X = df_encoded.drop(['sex', 'income'], axis=1)
    y_orig = df_encoded['income']

    # encode s
    s = []
    for j in s_orig:
        if "Female" in j:
            s.append(1)
        else:
            s.append(0)

    if intersectional:
        s_additional = X['race']
        X = X.drop(['race'], axis=1)
        s_additional = s_additional.astype(str)
        s_orig = [str(si) for si in s]
        unprivileged_group = 'Black_1'  # this is an hypothesis
        s = s_additional + '_' + s_orig
    else:
        unprivileged_group = 0

    pos_outcome = 1  # " >50K"

    y = []
    for k in y_orig:
        if '<=50K' in k:
            y.append(0)
        else:
            y.append(1)

    return X, pd.Series(y), s, unprivileged_group, pos_outcome


def get_dutch_census(url=os.path.join(PROJECT_ROOT, 'data', 'dutch_census_2001.csv'), intersectional=False):
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

    s_orig = (s_orig == 2)

    if intersectional:
        categorical_features.remove('citizenship')
        s_add = (df['citizenship'] == 1).astype(int).astype(str)
        s_orig = s_orig.astype(int).astype(str)
        s_orig = s_orig + '_' + s_add
        s_orig = pd.DataFrame(s_orig,
                              columns=['iswoman_iscitizen'])  # conversion due to error in train_test_split
        unprivileged_group = '1_0'
        df = df.drop(['citizenship'], axis=1)
    else:
        unprivileged_group = 1

    df_encoded = pd.get_dummies(df, columns=categorical_features)

    # Separate features and target variable
    X = df_encoded.drop(['sex', 'occupation'], axis=1)
    y_orig = df_encoded['occupation']

    # Encode y as 0/1
    y = []
    for target in y_orig:
        if '2_1' in target:
            y.append(1)  # high-level occupation (1: legislators, senior officials, managers; 2: professionals)
        else:
            y.append(0)  # '5_4_9' is low-level occupation (5: service workers, 4: clerks, 9: elementary occupations)

    pos_outcome = 1

    return X, pd.Series(y), s_orig, unprivileged_group, pos_outcome


def get_german(file=os.path.join(PROJECT_ROOT, 'data', 'german-credit.csv'), intersectional=False):
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
    s_orig = [0 if age <= 25 else 1 for age in s_orig]

    if intersectional:
        categorical_features.remove('personal_status_sex')
        s_add = df['personal_status_sex']
        s_orig = np.array(s_orig).astype(str)
        s_orig = s_orig + '_' + s_add
        s_orig = s_orig.rename('isOlderThan25_personal_status_sex')
        s_orig = s_orig.to_frame()
        unprivileged_group = '0_Female : single'  # guesswork
        print(s_orig)
        df = df.drop(['personal_status_sex'], axis=1)

    else:
        unprivileged_group = 0

    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=categorical_features)

    # Separate features and target variable
    y_orig = df_encoded['credit_risk']
    X = df_encoded.drop(['age', 'credit_risk'], axis=1)

    # Encode 'y' (target variable) as 0 (bad credit risk) or 1 (good credit risk)
    y = [0 if val == 0 else 1 for val in y_orig]

    # Define unprivileged group and positive outcome
    unprivileged_group = 0  # Young means unprivileged
    pos_outcome = 1  # Good credit risk

    return X, pd.Series(y), s_orig, unprivileged_group, pos_outcome


def get_folktables(state='AK', file=os.path.join(PROJECT_ROOT, 'data', 'folktables_{}_Income_2017.csv'),
                   intersectional=False):
    file = file.format(state)
    df = pd.read_csv(file, sep=',', header=0, index_col=0)
    column_names = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "PINCP", "RAC1P"]

    categorical_features = ["COW", "MAR", "POBP", "RELP", "SEX"]
    numerical_features = ["AGEP", "OCCP", "WKHP"]  # OCCP is categorical but unfeasible to one-hot
    ordinal_features = ["SCHL"]

    df_encoded = pd.get_dummies(df, columns=categorical_features)

    y = df_encoded['PINCP']  # income
    X = df_encoded.drop(['RAC1P', 'PINCP'], axis=1)

    # Binary race: 1 = white, 0 = non-white
    race = df_encoded['RAC1P']

    if intersectional:
        sex = df['SEX'].astype(int)  # 1 = male, 2 = female
        sex = (sex == 1).astype(int)  # now: 1 = male, 0 = female

        # Combine to 4 groups
        s = race.astype(str) + "_" + sex.astype(str)
        s = s.rename("race_sex")

        unprivileged_group = "0_0"  # non-white female
    else:
        s = race
        unprivileged_group = 0  # non-white

    pos_outcome = 1  # high income

    return X, pd.Series(y), list(s), unprivileged_group, pos_outcome


def get_banks(intersectional=False):
    column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                    'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'bank.csv'), delimiter=';')

    s_orig = df['age']

    numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

    # One-hot encode categorical features
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

    s = []
    for age in s_orig:
        if int(age) < 25:
            s.append(0)  # Young
        else:
            s.append(1)  # Old

    if intersectional:
        categorical_features.remove('marital')
        s_add = df['marital']
        s_orig = np.array([str(si) for si in s])
        s_orig = s_orig + '_' + s_add
        s_orig = s_orig.rename('isOlderThan25_marital')
        s_orig = s_orig.to_frame()
        unprivileged_group = '0_Divorced'  # only guesswork
        df = df.drop(['marital'], axis=1)
    else:
        unprivileged_group = 0

    df_encoded = pd.get_dummies(df, columns=categorical_features)
    # Separate features and target variable
    X = df_encoded.drop(['age', 'y'], axis=1)
    y_orig = df_encoded['y']

    # Encode y as 0/1
    y = []
    for target in y_orig:
        if 'no' in target:
            y.append(0)
        else:
            y.append(1)

    pos_outcome = 1  # "yes"

    return X, pd.Series(y), s_orig, unprivileged_group, pos_outcome


def get_law(url=os.path.join(PROJECT_ROOT, 'data', 'law_dataset.csv'), intersectional=False):
    column_names = ['decile1b', 'decile3', 'lsat', 'ugpa', 'zfygpa', 'zgpa', 'fulltime', 'fam_inc', 'male', 'race',
                    'tier', 'pass_bar']
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', url), delimiter=',')
    # Drop rows with missing values
    df = df.dropna().reset_index(drop=True)
    s_orig = df['race']
    numerical_features = ['decile1b', 'decile3', 'lsat', 'ugpa', 'zfygpa', 'zgpa']

    # One-hot encode categorical features
    categorical_features = ['fulltime', 'fam_inc', 'male', 'tier']

    if intersectional:
        categorical_features.remove('male')
        s_add = df['male'].astype(int).astype(str).to_numpy()
        s_orig = s_orig.astype(int).astype(str).to_numpy()
        s_orig = s_orig + '_' + s_add
        s_orig = pd.DataFrame(s_orig, columns=['black_female'])
        unprivileged_group = '0_0'
        df = df.drop(['male'], axis=1)
    else:
        unprivileged_group = 0

    df_encoded = pd.get_dummies(df, columns=categorical_features)

    # Separate features and target variable
    X = df_encoded.drop(['race', 'pass_bar'], axis=1)
    y_orig = df_encoded['pass_bar']
    pos_outcome = 1
    return X, pd.Series(y_orig), s_orig, unprivileged_group, pos_outcome

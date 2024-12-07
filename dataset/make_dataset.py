import numpy as np
import pandas as pd
from scipy.special import expit

np.random.seed(42)
df_complete = pd.read_csv('./dataset/original_compas.csv', index_col=None).dropna()
df_complete['sex'].replace({'Male': 1, 'Female': 0}, inplace=True)

def generate_mcar(df, features, missing_rate=0.1, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    mcar_data = df.copy()
    num_rows = df.shape[0]

    num_missing_rows = int(num_rows * missing_rate)
    mcar_indices = np.random.choice(num_rows, size=num_missing_rows, replace=False)

    for idx in mcar_indices:
        num_features_to_remove = np.random.randint(1, len(features) + 1)
        features_to_remove = np.random.choice(features, size=num_features_to_remove, replace=False)
        mcar_data.loc[idx, features_to_remove] = np.nan

    return mcar_data

def generate_mar(df, features, threshold=0.01, max_attempts=1000):
    num_rows = df.shape[0]
    attempt = 0

    while attempt < max_attempts:
        df_copy = df.copy()

        for feature in features:
            num_missing = np.random.randint(1, num_rows)
            missing_indices = np.random.choice(num_rows, size=num_missing, replace=False)
            df_copy.loc[missing_indices, feature] = np.nan

        observed_rows = ~df_copy[features].isna().any(axis=1)

        empirical_mean_observed = df_copy.loc[observed_rows, :].mean().mean()

        prob_mar = expit(0.1 * empirical_mean_observed)

        empirical_missing_rate = (~observed_rows).mean()

        if abs(empirical_missing_rate - prob_mar) < threshold:
            print(f"MAR achieved with empirical missing rate: {empirical_missing_rate:.3f} after {attempt + 1} attempts")
            return df_copy

        attempt += 1

    print("MAR mechanism not achieved within the maximum attempts.")
    return None


def generate_mnar(df, features, threshold=0.01, max_attempts=1000):
    num_rows = df.shape[0]
    attempt = 0

    while attempt < max_attempts:
        df_copy = df.copy()

        for feature in features:
            num_missing = np.random.randint(1, num_rows)
            missing_indices = np.random.choice(num_rows, size=num_missing, replace=False)
            df_copy.loc[missing_indices, feature] = np.nan

        missing_rows = df_copy[features].isna().any(axis=1)

        empirical_mean_missing = df_copy.loc[missing_rows, :].mean().mean()

        prob_mnar = expit(0.1 * empirical_mean_missing)

        empirical_missing_rate = missing_rows.mean()

        if abs(empirical_missing_rate - prob_mnar) < threshold:
            print(f"MNAR achieved with empirical missing rate: {empirical_missing_rate:.3f} after {attempt + 1} attempts")
            return df_copy

        attempt += 1

    print("MNAR mechanism not achieved within the maximum attempts.")
    return None

features=['age']
# features=['sex', 'age', 'priors_count', 'decile_score']
df_mcar = generate_mcar(df_complete, features)
# df_mar = generate_mar(df_complete, features)
# df_mnar = generate_mnar(df_complete, features)

df_mcar.to_csv("./dataset/mcar_compas.csv", index=None)
# print("MCAR Missing Values:\n", df_mcar.isna().sum())
# print("\nMAR Missing Values:\n", df_mar.isna().sum())
# print("\nMNAR Missing Values:\n", df_mnar.isna().sum())
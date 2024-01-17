import math

import numpy as np
import pandas as pd


def calculate_data_existing_length(data):
    res = 0
    for i in data:
        if not pd.isna(i):
            res += 1
    return res


# elements in data are sorted in time ascending order
def fill_missing_value(data, to_fill_value=0):
    data_len = len(data)
    data_exist_len = calculate_data_existing_length(data)
    if data_len == data_exist_len:
        return data
    elif data_exist_len == 0:
        # data = [to_fill_value for _ in range(data_len)]
        for i in range(data_len):
            data[i] = to_fill_value
        return data
    if pd.isna(data[0]):
        # find the first non-nan value's position
        not_na_pos = 0
        for i in range(data_len):
            if not pd.isna(data[i]):
                not_na_pos = i
                break
        # fill element before the first non-nan value with median
        for i in range(not_na_pos):
            data[i] = to_fill_value
    # fill element after the first non-nan value
    for i in range(1, data_len):
        if pd.isna(data[i]):
            data[i] = data[i - 1]
    return data


def forward_fill_pipeline(
    df: pd.DataFrame,
    default_fill: pd.DataFrame,
    demographic_features: list[str],
    labtest_features: list[str],
    target_features: list[str],
    require_impute_features: list[str],
):
    grouped = df.groupby("PatientID")

    all_x = []
    all_y = []
    all_pid = []
    all_record_times = []  # List to store record times for each patient
    all_missing_masks = []
    

    for name, group in grouped:
        sorted_group = group.sort_values(by=["RecordTime"], ascending=True)
        patient_x = []
        patient_y = []
        patient_record_times = []  # List to store record times for the current patient
        patient_missing_masks = pd.isna(sorted_group[labtest_features]).values.astype(int).tolist()

        for f in require_impute_features:
            # if the f is not in the default_fill, then default to -1
            if f not in default_fill: # these are normally categorical features
                to_fill_value = -1
            else:
                to_fill_value = default_fill[f]
            # take median patient as the default to-fill missing value
            fill_missing_value(sorted_group[f].values, to_fill_value)

        for _, v in sorted_group.iterrows():
            patient_record_times.append(v['RecordTime'])

            target_values = []
            for f in target_features:
                target_values.append(v[f])
            patient_y.append(target_values)
            x = []
            for f in demographic_features + labtest_features:
                x.append(v[f])
            patient_x.append(x)
        all_x.append(patient_x)
        all_y.append(patient_y)
        all_pid.append(name)
        all_record_times.append(patient_record_times)
        all_missing_masks.append(patient_missing_masks)
    return all_x, all_y, all_pid, all_record_times, all_missing_masks


# outlier processing
def filter_outlier(element):
    if np.abs(float(element)) > 1e4:
        return 0
    else:
        return element

def normalize_dataframe(train_df, val_df, test_df, normalize_features, require_norm_later=True):
    # Calculate the quantiles
    q_low = train_df[normalize_features].quantile(0.05)
    q_high = train_df[normalize_features].quantile(0.95)

    # Filter the DataFrame based on the quantiles
    filtered_df = train_df[(train_df[normalize_features] > q_low) & (
        train_df[normalize_features] < q_high)]

    # Calculate the mean and standard deviation and median of the filtered data, also the default fill value
    train_mean = filtered_df[normalize_features].mean()
    train_std = filtered_df[normalize_features].std()
    train_median = filtered_df[normalize_features].median()

    # if certain feature's mean/std/median is NaN, then set it as 0. This feature will be filled with 0 in the following steps
    train_mean = train_mean.fillna(0)
    train_std = train_std.fillna(0)
    train_median = train_median.fillna(0)

    if require_norm_later:
        default_fill: pd.DataFrame = (train_median-train_mean)/(train_std+1e-12)
        # LOS info
        los_info = {"los_mean": train_mean["LOS"].item(
        ), "los_std": train_std["LOS"].item(), "los_median": train_median["LOS"].item()}

        # Calculate large los and threshold (optional, designed for covid-19 benchmark)
        los_array = train_df.groupby('PatientID')['LOS'].max().values
        los_p95 = np.percentile(los_array, 95)
        los_p5 = np.percentile(los_array, 5)
        filtered_los = los_array[(los_array >= los_p5) & (los_array <= los_p95)]
        los_info.update({"large_los": los_p95.item(), "threshold": filtered_los.mean().item()*0.5})


        # Z-score normalize the train, val, and test sets with train_mean and train_std
        train_df.loc[:, normalize_features] = (train_df.loc[:, normalize_features] - train_mean) / (train_std+1e-12)
        val_df.loc[:, normalize_features] = (val_df.loc[:, normalize_features] - train_mean) / (train_std+1e-12)
        test_df.loc[:, normalize_features] = (test_df.loc[:, normalize_features] - train_mean) / (train_std+1e-12)

        train_df.loc[:, normalize_features] = train_df.loc[:, normalize_features].map(filter_outlier)
        val_df.loc[:, normalize_features] = val_df.loc[:, normalize_features].map(filter_outlier)
        test_df.loc[:, normalize_features] = test_df.loc[:, normalize_features].map(filter_outlier)

        return train_df, val_df, test_df, default_fill, los_info, train_mean, train_std

    else:
        default_fill: pd.DataFrame = train_median
        return default_fill

def normalize_df_with_statistics(df, normalize_features, train_mean, train_std):
    df.loc[:, normalize_features] = (df.loc[:, normalize_features] - train_mean) / (train_std+1e-12)
    df.loc[:, normalize_features] = df.loc[:, normalize_features].map(filter_outlier)
    return df


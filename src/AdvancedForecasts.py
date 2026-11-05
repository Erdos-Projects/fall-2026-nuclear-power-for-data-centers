import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import itertools as itr

import xgboost
from sklearn.metrics import mean_absolute_error as mae

from tqdm.auto import tqdm

# All paths are relative to the repo, so the code runs from any clone
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

# Our cleaned reactor data lives here
cleaned_dir = DATA_DIR / "nrc" / "cleaned_data"

# The test period (2023-2025) starts here
TRAINING_PERIOD = pd.Timestamp("2023-01-01")

# Cutoffs for the cross validation folds in the hyperparameter search
CUTOFFS = pd.to_datetime([ "2015-01-01", "2017-01-01", "2019-01-01",  "2021-01-01"])

# Everything here is known at the start of the quarter being predicted
FEATURE_COLUMNS = [ "Unit", "current_quarter", "average_power_prev_quarter",  "quarters_since_last_refuel", "last_refuel_duration", "seasonal_naive", "power_2d_before"]
TARGET = "average_power"

def load_daily_data():

    all_years=[]

    for year in range(2000, 2026):
        yearly_df = pd.read_csv(cleaned_dir/f'CleanedData_{year}.csv', sep="\t", parse_dates=["date"])
        all_years.append(yearly_df)

    # pd.concat requires a list of all dataframes
    df = pd.concat(all_years, ignore_index=True)

    df = df.sort_values(["Unit", "date"]).reset_index(drop=True)

    return df

def build_quarterly(df):

    # Now we will construct the quarterly data like we did for the preliminary forecasts
    df["quarter_period"] = df["date"].dt.to_period("Q")
    quarter_df = df.groupby(["Unit", "Region", "quarter_period"])["Power"].agg(["mean"]).reset_index()
    quarter_df = quarter_df.rename(columns={"mean": "average_power"})

    quarter_df["quarter_start"] = quarter_df["quarter_period"].dt.start_time
    quarter_df["current_quarter"] = quarter_df["quarter_period"].dt.quarter
    quarter_df = quarter_df.drop(columns=["quarter_period"])

    quarter_df = quarter_df.sort_values(["Unit", "quarter_start"]).reset_index(drop=True)

    # Now in order to build the feature: average_power_prev_quarter we can use the shift() operator
    # but in order for it to work properly we need to group by unit and sort according to the quarter start
    quarter_df["average_power_prev_quarter"] = quarter_df.groupby("Unit")["average_power"].shift(1)

    return quarter_df

def load_refueling():

    # Load the outages data and get the rows corresponding to refueling outages
    outages_df = pd.read_csv(cleaned_dir/"outages_data.csv", sep="\t", parse_dates=["Start_Date", "End_Date"])
    refueling_df = outages_df[ outages_df["outage_reason"] == "refueling outage" ][["Unit", "Start_Date", "End_Date", "Outage_Duration"]]
    refueling_df = refueling_df.sort_values(["Unit", "Start_Date"]).reset_index(drop=True)
    refueling_df["quarters_since_last_refuel"] = (refueling_df["Start_Date"] - refueling_df.groupby("Unit")["End_Date"].shift(1)).dt.days / 91

    return refueling_df

def add_refuel_features(quarter_df, refueling_df):

    # For every quarter, find the last refuel of the same reactor that ended before the quarter started
    quarter_df = pd.merge_asof(quarter_df.sort_values(["quarter_start"]), refueling_df.sort_values(["End_Date"]), left_on="quarter_start", right_on="End_Date", by="Unit", direction="backward", allow_exact_matches=False)

    # quarters are usually 90-92 days so dividing by the average
    # This overwrites the quarters_since_last_refuel column that came from refueling_df with the time since that refuel ended
    quarter_df["quarters_since_last_refuel"] = (quarter_df["quarter_start"] - quarter_df["End_Date"]).dt.days / 91
    quarter_df = quarter_df.rename(columns={"Outage_Duration": "last_refuel_duration"})
    quarter_df = quarter_df.drop(columns=["Start_Date", "End_Date"])

    quarter_df = quarter_df.sort_values(["Unit", "quarter_start"]).reset_index(drop=True)

    return quarter_df

def get_cycle_quarters(refueling_df, unit, date, n_recent=3):

    # Cycle length (6 or 8 quarters) as known at the given date, same idea as in PreliminaryForecasts
    mask =  (refueling_df["Unit"] == unit) & (refueling_df["End_Date"] < date) & (refueling_df["quarters_since_last_refuel"] > 3)
    cycle = refueling_df.loc[ mask, "quarters_since_last_refuel"].tail(n_recent)

    if len(cycle) == 0:
        return 0

    return 6 if cycle.median() < 7 else 8

def add_power_features(quarter_df, df, refueling_df):

    # Now the two remaining features are: seasonal naive forecast for each quarter and the average_power a few days before the quarter started
    # We will use the average power 2 days before the quarter start

    # This will make the search faster
    by_unit = {unit: unit_df for unit, unit_df in df.groupby("Unit")}
    quarterly_by_unit = {unit: unit_df for unit, unit_df in quarter_df.groupby("Unit")}

    power_2d_before=[]
    seasonal_naive=[]

    for unit, quarter_start in zip(quarter_df["Unit"], quarter_df["quarter_start"]):

        unit_df = by_unit[unit]
        unit_quarterly_df = quarterly_by_unit[unit]

        mask = (unit_df["date"] >= quarter_start - pd.Timedelta(days=2)) & (unit_df["date"] < quarter_start)
        power_2d_before.append(unit_df.loc[mask, "Power"].mean())

        cycle_length = get_cycle_quarters(refueling_df, unit, quarter_start)

        if cycle_length == 0:
            seasonal_naive.append(np.nan)
            continue

        mask_seasonal = unit_quarterly_df["quarter_start"] == (quarter_start - pd.DateOffset(months=3*cycle_length))
        naive = unit_quarterly_df.loc[mask_seasonal, "average_power"].mean()
        seasonal_naive.append(naive)

    quarter_df["power_2d_before"] = power_2d_before
    quarter_df["seasonal_naive"] = seasonal_naive

    # We do not need to remove rows with NaN cells as XGBoost handles them internally
    return quarter_df

def split_data(quarter_df):

    # unit will be used as a categorical variable so we need to define it as is
    quarter_df["Unit"] = quarter_df["Unit"].astype("category")

    # we need to remove Vogtle 3 and Vogtle 4 from consideration as we do not have enough data for these reactors
    # They started around 2024
    quarter_df = quarter_df[ ~quarter_df["Unit"].isin(["Vogtle 3", "Vogtle 4"])  ].reset_index(drop=True)

    # Watts Bar 2 began commercial operation in late 2016; earlier quarters are the commissioning ramp, not normal operation.
    commissioning = (quarter_df["Unit"] == "Watts Bar 2") & (quarter_df["quarter_start"] < "2016-10-01")
    quarter_df = quarter_df[~commissioning]

    training_df = quarter_df[ quarter_df["quarter_start"] < TRAINING_PERIOD ].reset_index(drop=True)
    testing_df = quarter_df[ quarter_df["quarter_start"] >= TRAINING_PERIOD ].reset_index(drop=True)

    return training_df, testing_df

def tune_xgboost(training_df):

    # The hyperparameters we are interested in are learning_rate, max_depth and n_estimators
    # For max_depth and learning_rate we perform a grid search with cross validation folds
    # For n_estimators we can put an arbitrarily large number for each combination and use early stopping
    # so the algorithm decides how many n_estimators are needed for each combination

    param_grid = { "max_depth": [2,3,5,10,20], "learning_rate": [0.001, 0.01, 0.1] }

    combinations = list(itr.product(*param_grid.values()))

    scores=[]

    for value in tqdm(combinations, desc="XGBoost grid search"):

        params = dict( zip(param_grid.keys(), value) )
        fold_scores=[]
        n_iterations=[]

        for cut_off in CUTOFFS:

            fold_df = training_df[ training_df["quarter_start"] < cut_off ].reset_index(drop=True)
            val_df = training_df[ training_df["quarter_start"] >= cut_off ]

            val_df = val_df[ val_df["Unit"].isin(fold_df["Unit"].unique()) ].reset_index(drop=True)

            model = xgboost.XGBRegressor(objective="reg:absoluteerror", enable_categorical=True, n_estimators=1000, early_stopping_rounds=50, eval_metric = "mae", **params)
            model.fit( fold_df[FEATURE_COLUMNS], fold_df[TARGET], eval_set=[(val_df[FEATURE_COLUMNS], val_df[TARGET])], verbose=False )
            fold_scores.append( mae(val_df[TARGET], model.predict(val_df[FEATURE_COLUMNS])) )
            n_iterations.append( model.best_iteration )

        scores.append({**params, "n_estimators": np.mean(n_iterations), "MAE":np.mean(fold_scores), "MAE_STD":np.std(fold_scores)})

    scores = pd.DataFrame(scores)
    scores = scores.sort_values(["MAE"]).reset_index(drop=True)

    print("\nBest hyperparameter combinations (cross validation MAE):")
    print(scores.head())
    print(" ")

    return scores

def train_model(training_df, scores):

    # The best combination is the first row, since scores is sorted by MAE
    best_n_estimators = scores["n_estimators"].iloc[0]
    best_max_depth = scores["max_depth"].iloc[0]
    best_learning_rate = scores["learning_rate"].iloc[0]

    model = xgboost.XGBRegressor(objective="reg:absoluteerror", enable_categorical=True, n_estimators=int(best_n_estimators), max_depth=best_max_depth, learning_rate=best_learning_rate)
    model.fit(training_df[FEATURE_COLUMNS], training_df[TARGET])

    return model

def test_model(model, testing_df):

    # Now we can test on the testing_df separated by reactor
    model_scores = []

    # observed=True skips the reactors we removed (Vogtle 3 and 4), which are still listed in the Unit categories
    for unit, unit_df in testing_df.groupby("Unit", observed=True):

        pred = model.predict(unit_df[FEATURE_COLUMNS])
        model_scores.append( {"Unit": unit, "Region": unit_df["Region"].iloc[0], "MAE": mae(unit_df[TARGET], pred)} )
        testing_df.loc[unit_df.index, "xgb_pred"] = pred

    model_scores = pd.DataFrame(model_scores)

    print("Reactors with the largest test errors:")
    print(model_scores.sort_values("MAE").tail())
    print(" ")

    return testing_df, model_scores

def plot_test_scores(model_scores):

    model_scores.boxplot(column=["MAE"], by="Region", figsize=(5,5))
    plt.ylabel("MAE")
    plt.ylim(0, 40)
    plt.show()

def plot_forecasts(training_df, testing_df, region):

    regional_df = testing_df[ testing_df["Region"] == region ]
    reactors = regional_df["Unit"].unique()

    n_rows = 5
    n_columns = math.ceil(len(reactors)/n_rows)

    fig, axes = plt.subplots(n_rows, n_columns, figsize=(15,10))
    axes = axes.flatten()

    for i in range(len(reactors)):

        train_mask = training_df["Unit"] == reactors[i]
        test_mask = regional_df["Unit"] == reactors[i]

        axes[i].plot(training_df[train_mask]["quarter_start"], training_df[train_mask]["average_power"], label='Training Data')
        axes[i].plot(regional_df[test_mask]["quarter_start"], regional_df[test_mask]["average_power"], label='Test Data', color='red')
        axes[i].plot(regional_df[test_mask]["quarter_start"], regional_df[test_mask]["xgb_pred"], "--", label='XGBoost', color='blue')

        axes[i].set_xlabel("Date")
        axes[i].set_ylabel("Average Power")
        axes[i].set_title(f'{reactors[i]}')
        axes[i].legend(frameon=False)
        axes[i].set_xlim(pd.Timestamp("2010-01-01"), pd.Timestamp("2026-01-01"))
        axes[i].set_ylim(0, 110)

        axes[i].xaxis.set_major_locator(mdates.YearLocator(5))

    # Hide the empty plots
    for j in range(len(reactors), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def summarize(model_scores):

    summary = model_scores["MAE"].describe(percentiles=[0.25, 0.5, 0.75, 0.9])

    print("Test MAE per reactor over 2023-2025:")
    print(summary.round(2))
    print(" ")

    # XGBoost is the most accurate model we have tried. Compared with the seasonal average from PreliminaryForecasts,
    # it lowers the mean, the median and the 90th percentile error. The hardest reactors are still hard:
    # the worst reactor's error barely changes.

    return summary

def bootstrap_improvement(model_scores, n_resamples=20000, seed=2026):

    # Compare against the seasonal average scores saved by PreliminaryForecasts.py
    baseline = pd.read_csv(DATA_DIR / "seasonal_average_scores.csv")
    merged = baseline.merge(model_scores[["Unit", "MAE"]].astype({"Unit": str}), on="Unit")

    baseline_mae = merged["seasonal_average"].to_numpy()
    xgb_mae = merged["MAE"].to_numpy()

    # Resample the reactors with replacement. Each row of idx is one resample.
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(merged), (n_resamples, len(merged)))

    improvement = 1 - xgb_mae[idx].mean(axis=1) / baseline_mae[idx].mean(axis=1)
    low, high = np.percentile(improvement, [2.5, 97.5])

    print(f"XGBoost improvement over seasonal average: {1 - xgb_mae.mean() / baseline_mae.mean():.1%}")
    print(f"95% bootstrap interval: {low:.1%} to {high:.1%}")

    return low, high

def main():

    df = load_daily_data()
    quarter_df = build_quarterly(df)

    refueling_df = load_refueling()
    quarter_df = add_refuel_features(quarter_df, refueling_df)
    quarter_df = add_power_features(quarter_df, df, refueling_df)

    training_df, testing_df = split_data(quarter_df)

    scores = tune_xgboost(training_df)
    model = train_model(training_df, scores)

    testing_df, model_scores = test_model(model, testing_df)
    plot_test_scores(model_scores)
    plot_forecasts(training_df, testing_df, "Region 1")

    summarize(model_scores)
    bootstrap_improvement(model_scores)

if __name__ == "__main__":
    main()

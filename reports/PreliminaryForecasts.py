import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import itertools as itr

import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing
from pmdarima import auto_arima, ARIMA
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_error as mae

from tqdm.auto import tqdm

# Change DATA_DIR to wherever the reactor_data folder is on your computer
DATA_DIR = Path("/Users/vassu/Desktop/Erdos_DataScience/NuclearReliability/reactor_data")

# Our cleaned reactor data lives here
cleaned_dir = DATA_DIR / "cleaned_data"

# Tuning period cutoffs (2000-2022) used to choose each model's settings
CUTOFFS = pd.to_datetime(["2009-01-01", "2011-01-01", "2013-01-01", "2015-01-01", "2017-01-01" , "2020-01-01"])

# The test period (2023-2025) starts here
TRAINING_PERIOD = pd.Timestamp("2023-01-01")

METHODS = ["seasonal_naive", "seasonal_average", "holt_winter", "SARIMA"]

def load_quarterly_data():

    # load the data. We need to pass on the parse_dates option so that we can perform operations on the date column
    # By default the date column will be treated as a string (because of the dashes)
    all_years = []

    for year in range(2000, 2026):

        df = pd.read_csv(cleaned_dir/f"CleanedData_{year}.csv", sep="\t", parse_dates=["date"])
        all_years.append(df)

    df = pd.concat(all_years, ignore_index=True)

    # Using the dates Column to define a quarter period
    df["quarter_period"] = df["date"].dt.to_period("Q")

    # groupby turns Unit, Region and quarter_period into the index, with the mean as the only column.
    # reset_index moves them back into regular columns.
    quarter_df = df.groupby(["Unit", "Region", "quarter_period"])["Power"].agg(["mean"]).reset_index()
    quarter_df = quarter_df.rename(columns={"mean": "average_power"})

    quarter_df["quarter_start"] = quarter_df["quarter_period"].dt.start_time

    # We drop the Watts Bar 2 quarters before October 2016 (its commissioning period, not normal operation)
    # and leave out Vogtle 3 and 4, which started operating too recently (not enough data).
    commissioning = (quarter_df["Unit"] == "Watts Bar 2") & (quarter_df["quarter_start"] < "2016-10-01")
    quarter_df = quarter_df[~commissioning]

    quarter_df = quarter_df[ ~quarter_df["Unit"].isin(["Vogtle 3", "Vogtle 4"])  ].reset_index(drop=True)

    print("We forecast the quarterly power of", quarter_df["Unit"].nunique(), "reactors")

    return quarter_df

def plot_quarterly_power(quarter_df, region):

    # Things to look for: a trend, repeating patterns (regular drops) and one-off events
    regional_df = quarter_df[ quarter_df["Region"] == region ]
    reactors = regional_df["Unit"].unique()

    n_rows = 4
    n_columns = math.ceil(len(reactors)/n_rows)

    fig, axes = plt.subplots(n_rows, n_columns, figsize=(15,10))
    axes = axes.flatten()

    for i in range(len(reactors)):
        axes[i].plot(regional_df[regional_df["Unit"] == reactors[i]]["quarter_start"], regional_df[regional_df["Unit"] == reactors[i]]["average_power"])
        axes[i].set_xlabel("Year")
        axes[i].set_ylabel("Average Power")
        axes[i].set_title(f'{reactors[i]}')
        axes[i].xaxis.set_major_locator(mdates.YearLocator(5))

    # Hide the empty plots
    for j in range(len(reactors), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    # Most reactors run close to 100% power, with sharp drops at regular intervals. These are the refueling outages.
    # There is no long-term trend, but the series is clearly seasonal, and the season length differs between reactors.

def load_refueling():

    outages_df = pd.read_csv(cleaned_dir/"outages_data.csv", sep="\t", parse_dates=["Start_Date", "End_Date"])

    refueling_df = outages_df[ outages_df["outage_reason"] == "refueling outage" ].sort_values(["Unit", "Start_Date"]).reset_index(drop=True)

    # For each refueling outage: time from the end of the previous
    # refuel to the start of this one, in quarters (91 days ~ 1 quarter).
    # shift(1) within each Unit gives the previous refuel of the same reactor
    # A reactor's first refuel has no previous one, so it gets NaN.
    refueling_df["quarters_since_last_refuel"] = (refueling_df["Start_Date"] - refueling_df.groupby("Unit")["End_Date"].shift(1)).dt.days/(91)

    return refueling_df

def plot_refuel_intervals(refueling_df, region):

    regional_df = refueling_df[ refueling_df["Region"] == region ]
    reactors = regional_df["Unit"].unique()

    n_rows = 5
    n_columns = math.ceil(len(reactors)/n_rows)

    fig, axes = plt.subplots(n_rows, n_columns, figsize=(12,8))
    axes = axes.flatten()

    for i in range(len(reactors)):

        axes[i].hist(regional_df[ regional_df["Unit"] == reactors[i]]["quarters_since_last_refuel"], bins=16, range=(0, 10))
        axes[i].set_xlabel("Quarters between refuel")
        axes[i].set_ylabel("Counts")
        axes[i].set_title(f'{reactors[i]}')

    # Hide the empty plots
    for j in range(len(reactors), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    # Most reactors fall into an 18-month or a 24-month refueling cycle, but many are a mix of the two.
    # We measure from the end of one refuel to the start of the next, which leaves out the outage itself,
    # so the histograms peak a little below 6 and 8 quarters.

def get_cycle_quarters(refueling_df, unit, cutoff, n_recent=3):

    # Cycle length (6 or 8 quarters) of a reactor, as known at a given date (cutoff).
    # It is called with the start date of every quarter we forecast,
    # so it only uses refuels that had ENDED before that date: no information from the future.
    #
    # Intervals shorter than 4 quarters are mid-cycle outages, not real cycles, so they are skipped.
    # We take the median of the last n_recent intervals, so the cycle follows recent changes but one
    # odd interval doesn't dominate. The 7-quarter threshold sits between the two groups
    # (our intervals exclude the outage itself, so they come out slightly under 6 and 8).
    # Returns 0 if the reactor has no usable refuel history yet, meaning "no forecast possible".

    mask = (refueling_df["End_Date"] < cutoff) & (refueling_df["quarters_since_last_refuel"] >= 4) & (refueling_df["Unit"] == unit)
    past = refueling_df[mask]

    if len(past) == 0:
        return 0

    length = past["quarters_since_last_refuel"].tail(n_recent).median()
    return 6 if length < 7 else 8

def choose_k(quarter_df, refueling_df):

    cycles_considered = [2, 3, 4, 5]
    scores=[]

    # Choosing k (the number of past cycles to average) for the seasonal average.
    # Nothing is fitted here: for each candidate k we make walk-forward, one-quarter-ahead forecasts
    # over the tuning period (defined in the cut offs) and compare the errors.

    for unit, unit_df in quarter_df.groupby("Unit"):

        y = unit_df["average_power"].to_numpy()

        for i in range(len(CUTOFFS)-1):

            mask = (unit_df["quarter_start"] >= CUTOFFS[i]) & (unit_df["quarter_start"] < CUTOFFS[i+1])
            training_data = unit_df[mask]

            # Position of the window's first quarter in y (summing True/False counts the quarters before the window)
            start = (unit_df["quarter_start"] < CUTOFFS[i]).sum()

            # A reactor might not have data for the cutoff period considered
            # Putting a safeguard for that
            if len(training_data) == 0:
                continue

            for cycle in cycles_considered:

                actual = []
                forecast = []

                for j, quarter_start in enumerate(training_data["quarter_start"]):

                    # Cycle length as known at the start of this quarter (no future refuels)
                    # Our walk forward forecast idea
                    length = get_cycle_quarters(refueling_df, unit, quarter_start)

                    if (length == 0):
                        continue

                    # j+start is the position of this quarter in the y array
                    cycles_available = (j+start) //  length
                    k = min(cycle, cycles_available)

                    # Without this safeguard vals would be empty and np.mean([]) would return NaN
                    if k == 0:
                        continue

                    vals = [ y[ (j+start) - i*length] for i in range(1, k+1)]
                    forecast.append(np.mean(vals))
                    actual.append( y[j+start])

                # This is important because an empty forecast/actual array passed into the mae/rmse
                # function will raise an error (annoying)
                if len(actual) == 0:
                    continue

                scores.append({"Unit": unit, "cut_off_start": CUTOFFS[i], "cut_off_end": CUTOFFS[i+1], "cycles": cycle, "RMSE": rmse(actual, forecast), "MAE": mae(actual, forecast) })

    scores = pd.DataFrame(scores)

    print("\nMean error of the seasonal average over the tuning period for each cycle choice:")
    print(scores.groupby(["cycles"])[["RMSE", "MAE"]].agg(["mean"]))
    print(" ")

    # All values of k give very similar errors. We choose k = 3, which has the lowest mean MAE.
    # We rely on MAE rather than RMSE: RMSE squares the errors, so it is dominated by a few very large misses
    # (unplanned or unusually long outages) that no seasonal model can predict.

    return scores

def baseline_forecasts(quarter_df, refueling_df, cycles_considered=3):

    for unit, unit_df in quarter_df.groupby("Unit"):

        y = unit_df["average_power"].to_numpy()
        seasonal_naive = []
        seasonal_average = []

        # For a walk forward forecast we need to go row by row
        # quarter_start basically helps us define the cycle length
        # using refuel data before this date
        for j, quarter_start in enumerate(unit_df["quarter_start"]):

            length = get_cycle_quarters(refueling_df, unit, quarter_start)

            # if the forecast is not possible because there is not enough
            # refueling history yet or not even one full cycle of data before this quarter
            if (length == 0) or (j-length < 0):
                seasonal_naive.append(np.nan)
                seasonal_average.append(np.nan)
                continue

            # The value of this quarter in the previous cycle
            seasonal_naive.append(y[j-length])

            # Depending on where we are, the cycles available might be smaller than required for forecast
            cycles_available = j // length
            k = min(cycles_considered, cycles_available)

            if k == 0:
                seasonal_average.append(np.nan)
                continue

            vals = [ y[j-i*length] for i in range(1, k+1) ]
            seasonal_average.append(np.mean(vals))

        quarter_df.loc[ unit_df.index, "seasonal_naive" ] = seasonal_naive
        quarter_df.loc[ unit_df.index, "seasonal_average"] = seasonal_average

    return quarter_df

def tune_holt_winters(quarter_df, refueling_df):

    # For each reactor we try every pair of alpha and gamma. At each cutoff in the tuning period
    # we fit on the last 4 cycles before the cutoff, forecast the next full cycle and compute the MAE.
    # We keep the pair with the lowest average MAE across cutoffs.
    # In order to test against every combination of alphas and gammas we will use the "itertools" library

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    combinations = list(itr.product(alphas, gammas))

    results= {}

    for unit, unit_df in tqdm(quarter_df.groupby("Unit"), desc="Holt-Winters grid search"):

        best_score = np.inf

        for alpha, gamma in combinations:

            fold_scores = []

            for cut_off in CUTOFFS:

                length = get_cycle_quarters(refueling_df, unit, cut_off)

                if length == 0:
                    continue

                # Much easier to work with a numpy array
                train_set = unit_df[ unit_df["quarter_start"] < cut_off ]["average_power"].to_numpy()[-4*length:]
                val_set = unit_df[ unit_df["quarter_start"] >= cut_off ]["average_power"].to_numpy()[:length]

                # For initialization the model needs at least two full seasonal cycles
                # one cycle for initialization and one for the update
                # If there is nothing in the validation set that is also useless
                if len(train_set) < 2*length or len(val_set) == 0:
                    continue

                model = ExponentialSmoothing( train_set, seasonal='add', seasonal_periods=length).fit(smoothing_level=alpha, smoothing_seasonal=gamma, optimized=False)

                fold_scores.append(mae(val_set, model.forecast(len(val_set))))

            if len(fold_scores) == 0:
                continue

            score = np.mean(fold_scores)

            if score < best_score:
                best_score = score
                results[unit] = { "alpha": alpha, "gamma": gamma, "score": score }

    print("")
    return results

def holt_winters_forecasts(quarter_df, refueling_df, results):

    # Walk-forward: with alpha and gamma fixed, we refit the model for every quarter on the quarters before it and forecast that one quarter
    for unit, unit_df in quarter_df.groupby("Unit"):

        y = unit_df["average_power"].to_numpy()

        # Every reactor has tuned values (checked: no reactor is missing from results).
        alpha = results[unit]["alpha"]
        gamma = results[unit]["gamma"]

        holt_winter=[]

        for j, quarter_start in enumerate(unit_df["quarter_start"]):

            length = get_cycle_quarters(refueling_df, unit, quarter_start)

            if (length == 0) or (j < 2*length):
                holt_winter.append(np.nan)
                continue

            # only keeping the past 4 cycles
            model = ExponentialSmoothing(y[:j][-4*length:], seasonal='add', seasonal_periods=length).fit(smoothing_level=alpha, smoothing_seasonal=gamma, optimized=False)

            holt_winter.append(model.forecast(1)[0])

        quarter_df.loc[unit_df.index, "holt_winter"] = holt_winter

    return quarter_df

def plot_acf_pacf(quarter_df, region):

    # ACF: correlation between the series and itself shifted by k quarters
    # PACF: the same correlation after removing the effect of all shorter lags
    # Peaks at multiples of the cycle length confirm the refueling seasonality.
    # The shaded band marks the 95% significance level.

    regional_df = quarter_df[ quarter_df["Region"] == region ]
    reactors = regional_df["Unit"].unique()

    n_rows = 4
    n_columns = math.ceil(len(reactors)/n_rows)

    fig, axes = plt.subplots(n_rows, n_columns, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(len(reactors)):

        sm.graphics.tsa.plot_acf(regional_df[ regional_df["Unit"] == reactors[i] ]["average_power"].reset_index(drop=True), lags=50, ax=axes[i])

        axes[i].set_xlabel("Lag (In Quarters)")
        axes[i].set_ylabel("ACF")
        axes[i].set_title(f'{reactors[i]}')

    for j in range(len(reactors), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(n_rows, n_columns, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(len(reactors)):

        sm.graphics.tsa.plot_pacf(regional_df[ regional_df["Unit"] == reactors[i] ]["average_power"].reset_index(drop=True), lags=30, method = 'ols', ax=axes[i])

        axes[i].set_xlabel("Lag (In Quarters)")
        axes[i].set_ylabel("PACF")
        axes[i].set_title(f'{reactors[i]}')

    for j in range(len(reactors), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    # The ACF peaks at multiples of the cycle length are weak and fade quickly, and short-lag correlations are small.
    # Instead of choosing the orders by hand for every reactor, we let auto_arima search for them.

def sarima_forecasts(quarter_df, refueling_df):

    # For each reactor, auto_arima searches for the model orders on the last 4 refueling cycles before 2023
    # and picks the model with the lowest AIC. m = cycle length at the start of 2023, D = 1 (one seasonal difference), d = 0 (no trend)
    # The orders stay fixed in the test period, where we forecast one quarter and then add its actual value with model.update()

    sarima_models = {}

    for unit, unit_df in tqdm(quarter_df.groupby("Unit"), desc="SARIMA"):

        y = unit_df["average_power"].to_numpy()

        # number of training quarters, and the cycle length known at the start of the test period
        n_train = (unit_df["quarter_start"] < TRAINING_PERIOD).sum()
        length = get_cycle_quarters(refueling_df, unit, TRAINING_PERIOD)

        # Needs at least 2 cycles: one is used up by the seasonal difference,
        # at least one to estimate the coefficients
        if (length == 0) or (n_train < 2*length):
            continue

        start = max(0, n_train-4*length)

        # Choose the orders once, on the training quarters only (auto_arima compares models by AIC)
        best = auto_arima(y[start:n_train], seasonal=True, m=length, error_action="ignore", d=0, D=1)
        model = ARIMA(order=best.order, seasonal_order=best.seasonal_order, with_intercept=False, suppress_warnings=True).fit(y[start:n_train])
        sarima_models[unit] = {"order": best.order, "seasonal_order": best.seasonal_order, "aic": model.aic()}

        # Training quarters: one-quarter-ahead in-sample predictions (only for the plots, they are not true forecasts)
        in_sample = model.fittedvalues() # only holds (n_train-start) values
        arima = [np.nan]*start

        for j in range(n_train-start):
            if (j < length):
                arima.append(np.nan)
            else:
                arima.append(in_sample[j])

        # Test quarters, walk forward: forecast the next quarter, then add its actual value before moving on
        for actual in y[n_train:]:
            arima.append(model.predict(n_periods=1)[0])
            model.update([actual])

        quarter_df.loc[unit_df.index, "arima"] = arima

    return quarter_df, sarima_models

def score_models(quarter_df):

    all_scores={}

    for unit,unit_df in quarter_df.groupby("Unit"):

        # valid would be a series of True/False, False if any of the forecasts is empty or the quarter is not in the test period
        valid = (unit_df[["seasonal_naive", "seasonal_average", "holt_winter", "arima"]].notna().all(axis=1)) & (unit_df["quarter_start"] >= TRAINING_PERIOD)

        if valid.sum() == 0:
            continue

        naive = mae( unit_df.loc[valid, "average_power"], unit_df.loc[valid, "seasonal_naive"] )
        average = mae( unit_df.loc[valid, "average_power"], unit_df.loc[valid, "seasonal_average"] )
        holt = mae( unit_df.loc[valid, "average_power"], unit_df.loc[valid, "holt_winter"] )
        arima = mae( unit_df.loc[valid, "average_power"], unit_df.loc[valid, "arima"] )

        all_scores[unit] = { "Region": unit_df["Region"].iloc[0], "seasonal_naive": naive, "seasonal_average": average, "holt_winter":holt, "SARIMA":arima }

    all_scores = pd.DataFrame(all_scores).T
    all_scores = all_scores.reset_index()
    all_scores = all_scores.rename(columns={"index": "Unit"})

    all_scores[METHODS] = all_scores[METHODS].astype(float)

    return all_scores

def plot_scores(all_scores):

    region_mean = all_scores.groupby("Region")[METHODS].mean()

    region_mean.plot.bar(capsize=3, figsize=(10, 5))
    plt.ylabel("Test MAE")
    plt.xticks(rotation=0)
    plt.show()

    # Each box summarises the test MAE of all reactors in one region for one method.
    # Green line: the median reactor. Box: the middle 50% of reactors. Circles: single reactors with an unusually large error.
    axes = all_scores.boxplot(column=METHODS, by="Region", layout=(1, 4), figsize=(12, 4), rot=45, sharey=True)
    axes[0].set_ylabel("MAE")
    plt.ylim(0, 40)
    plt.tight_layout()
    plt.show()

def plot_forecasts(quarter_df, region):

    regional_df = quarter_df[ quarter_df["Region"] == region ]
    reactors = regional_df["Unit"].unique()

    n_rows = 4
    n_columns = math.ceil(len(reactors)/n_rows)

    fig, axes = plt.subplots(n_rows, n_columns, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(len(reactors)):

        reactor_df = regional_df[ regional_df["Unit"] == reactors[i] ]

        axes[i].plot( reactor_df["quarter_start"], reactor_df["average_power"], color='red', label="Data" )
        axes[i].plot( reactor_df["quarter_start"], reactor_df["seasonal_naive"], '--', color='black', label="Seasonal Naive" )
        axes[i].plot( reactor_df["quarter_start"], reactor_df["seasonal_average"], '--', color='green', label="Seasonal Average")
        axes[i].plot( reactor_df["quarter_start"], reactor_df["holt_winter"], '--', color='violet', label="Holt-Winters")
        axes[i].plot( reactor_df["quarter_start"], reactor_df["arima"], '--', color='cyan', label="SARIMA")

        axes[i].legend(fontsize=7, frameon=False)
        axes[i].set_xlabel("Year")
        axes[i].set_ylabel("Average Power")
        axes[i].set_title(f'{reactors[i]}')
        axes[i].set_ylim(-10, 110)
        axes[i].set_xlim(pd.Timestamp("2010-01-01"), pd.Timestamp("2026-01-01"))
        axes[i].xaxis.set_major_locator(mdates.YearLocator(5))

    for j in range(len(reactors), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def summarize(all_scores):

    # Summary of the per-reactor test MAE for each model (2023-2025, in percentage points of full power)
    summary = all_scores[METHODS].describe(percentiles=[0.25, 0.5, 0.75, 0.9]).T

    print("\nTest MAE per reactor over 2023-2025:")
    print(summary.round(2))
    print(" ")

    # The simple seasonal average (k = 3) is the best univariate model.
    # The errors are concentrated in a few reactors: all four models fail on unplanned outages
    # or refuelings that run much longer than usual, because none of that can be seen in past power alone.

    return summary

def main():

    quarter_df = load_quarterly_data()
    plot_quarterly_power(quarter_df, "Region 1")

    refueling_df = load_refueling()
    plot_refuel_intervals(refueling_df, "Region 1")

    choose_k(quarter_df, refueling_df)
    quarter_df = baseline_forecasts(quarter_df, refueling_df, cycles_considered=3)

    results = tune_holt_winters(quarter_df, refueling_df)
    quarter_df = holt_winters_forecasts(quarter_df, refueling_df, results)

    plot_acf_pacf(quarter_df, "Region 1")
    quarter_df, sarima_models = sarima_forecasts(quarter_df, refueling_df)

    all_scores = score_models(quarter_df)
    plot_scores(all_scores)
    plot_forecasts(quarter_df, "Region 1")

    summarize(all_scores)

if __name__ == "__main__":
    main()

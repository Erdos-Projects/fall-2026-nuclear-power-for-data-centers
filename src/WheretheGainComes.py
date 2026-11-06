"""Where does XGBoost's improvement over seasonal average come from?

Compares the two forecasts on the 2023-2025 test quarters (92 reactors, 12 quarters each).
A low-power quarter is one with average power below 90% of full power.
A false alarm is a forecast below 90% for a quarter that actually stayed at 90% or above.

Run from the src folder: python WhereTheGainComes.py
"""
import pandas as pd
import xgboost

import AdvancedForecasts as A
import PreliminaryForecasts as P

LOW_POWER = 90

# The tuned settings chosen by AdvancedForecasts.tune_xgboost on the training period
BEST = dict(n_estimators=296, max_depth=5, learning_rate=0.01)


def xgboost_forecasts():

    df = A.load_daily_data()
    refueling_df = A.load_refueling()
    quarter_df = A.build_quarterly(df)
    quarter_df = A.add_refuel_features(quarter_df, refueling_df)
    quarter_df = A.add_power_features(quarter_df, df, refueling_df)
    training_df, testing_df = A.split_data(quarter_df)

    model = xgboost.XGBRegressor(objective="reg:absoluteerror", enable_categorical=True, **BEST)
    model.fit(training_df[A.FEATURE_COLUMNS], training_df[A.TARGET])

    testing_df = testing_df.copy()
    testing_df["xgb_pred"] = model.predict(testing_df[A.FEATURE_COLUMNS])
    testing_df["Unit"] = testing_df["Unit"].astype(str)

    return testing_df[["Unit", "quarter_start", "average_power", "xgb_pred"]]


def seasonal_forecasts():

    quarter_df = P.load_quarterly_data(P.load_daily_data())
    forecasts = P.baseline_forecasts(quarter_df, P.load_refueling(), 3)
    forecasts["Unit"] = forecasts["Unit"].astype(str)

    return forecasts[["Unit", "quarter_start", "seasonal_average"]]


def summarize(test_df):

    actual = test_df["average_power"]
    low = actual < LOW_POWER
    rows = []

    for name, column in [("XGBoost", "xgb_pred"), ("Seasonal average", "seasonal_average")]:

        error = (test_df[column] - actual).abs()
        predicted_low = test_df[column] < LOW_POWER

        rows.append({
            "Model": name,
            "False alarms": int((predicted_low & ~low).sum()),
            "Low quarters flagged": int((predicted_low & low).sum()),
            "Error, low quarters": round(error[low].mean(), 2),
            "Error, normal quarters": round(error[~low].mean(), 2),
        })

    print(f"Quarters: {len(test_df)}  Reactors: {test_df['Unit'].nunique()}  "
          f"Normal: {int((~low).sum())}  Low-power: {int(low.sum())}\n")

    summary = pd.DataFrame(rows).set_index("Model")
    print(summary.T.to_string())

    return summary


def main():

    test_df = xgboost_forecasts().merge(seasonal_forecasts(), on=["Unit", "quarter_start"], how="inner")
    summarize(test_df)


if __name__ == "__main__":
    main()

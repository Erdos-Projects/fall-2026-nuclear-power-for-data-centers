import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math

from tqdm.auto import tqdm

from concurrent.futures import ThreadPoolExecutor
import time
import threading

# All paths are relative to the repo, so the code runs from any clone
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Downloaded NRC reports. data/raw is not committed to GitHub
daily_dir = DATA_DIR / "raw" / "nrc" / "daily_reports"
yearly_dir = DATA_DIR / "raw" / "nrc" / "yearly_reports"

# Cleaned outputs that PreliminaryForecasts and AdvancedForecasts read
cleaned_dir = DATA_DIR / "processed" / "nrc" / "cleaned_data"

URL="https://www.nrc.gov/documents-reports/document-collections/events-reports-associated-with/power-reactor-status-reports/"

# The same reactor is written differently in different years
UNIT_RENAMES = { "LaSalle 1": "La Salle 1", "LaSalle 2": "La Salle 2", "River Bend Station 1": "River Bend 1", "Arkansas Nuclear 1": "Arkansas Nuclear One 1", "Arkansas Nuclear 2": "Arkansas Nuclear One 2", "Summer": "V.C. Summer" }

thread_local = threading.local()

def get_session():

    # The first time a thread calls this, it has no session yet, so make one.
    # Every later call from that same thread reuses it.
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session

def get_daily_data(date):

    Full_URL = URL + f'{date.year}/{date:%Y%m%d}ps'

    webpage = None

    for attempt in range(3):

        # A timeout or dropped connection raises an error instead of returning a webpage, so we catch it and try again
        try:
            webpage = get_session().get(Full_URL, timeout=60)
            if webpage.ok:
                break
        except requests.RequestException:
            pass

        # No point waiting after the last attempt
        if attempt < 2:
            time.sleep(30)

    if webpage is None or not webpage.ok:
        print("Could not fetch data for", date.date())
        return None

    time.sleep(0.25)
    
    soup = BeautifulSoup(webpage.content, "html.parser")
    rows = []
    
    for table in soup.find_all("table"):

        # Each table sits right after a heading like <h3>Region 1</h3>
        heading = table.find_previous("h3")
        region = heading.get_text(strip=True) if heading else None

        for row in table.find_all("tr"):
    
            # Unit, Power, Down, Reason, Change in report, Scrams
            cells = [ td.get_text(strip=True) for td in row.find_all("td") ]
            if len(cells) != 6:
                continue
            rows.append({"date": date, "Region": region, "Unit": cells[0], "Power": cells[1], "Down": cells[2], "Reason": cells[3]})

    return rows

def generate_df():

    # An empty list to hold the names of the yearly csv files with daily reports
    report_list = []

    # The start is inclusive but stop is excluded
    for year in range(2000, 2026):

        outfile = daily_dir / f'{year}DailyReports.csv'

        if not outfile.exists():

            dates = pd.date_range(f'{year}-01-01', f'{year}-12-31')
            year_rows = []

            # get_daily_data returns a list of dicts [dict1, dict2 , ...]
            # daily_rows would be a list of list -- > [ [dict1, dict2], [dict3, dict4] ]
            with ThreadPoolExecutor(max_workers=2) as pool:
                daily_rows = list( tqdm(pool.map(get_daily_data, dates), total=len(dates), desc=str(year)))

            failed = sum(rows is None for rows in daily_rows)
            if failed > 0:
                print(f'{year}: {failed} days failed, not saving. Rerun the script later.')
                continue

            # This will basically give you [ dict1, dict2, dict3, dict4 ]
            for rows in daily_rows:
                year_rows.extend(rows)

            pd.DataFrame(year_rows).to_csv(outfile, index=False)

        report_list.append(pd.read_csv(outfile, parse_dates=["date"])) # Read the date column as a date not a text

    df = pd.concat(report_list, ignore_index=True)  
    return df

def initial_cleanup(df):

    df["Unit"] = df["Unit"].replace(UNIT_RENAMES)

    runtime_df = df.groupby("Unit")["date"].agg(["min", "max", "count"])
    runtime_filtered_df = runtime_df[ (runtime_df["min"] > "2000-01-01") | (runtime_df["max"] < "2025-12-31")]

    unoperating_reactors = runtime_filtered_df[ runtime_filtered_df["max"] < "2025-12-31" ].index
    new_df = df.drop(df[ df["Unit"].isin(unoperating_reactors) ].index).reset_index(drop=True)

    # Also need to remove the Palisades reactor that has some rows in year 2026 but it stopped operating in mid 2022.
    new_df = new_df.drop( new_df[new_df["Unit"] == "Palisades"].index ).reset_index(drop=True)

    # Removing any duplicates
    new_df = new_df.drop_duplicates(subset=["Unit", "date"], keep="first").reset_index(drop=True)

    new_df = new_df.sort_values(["Unit", "date"]).reset_index(drop=True)
    new_df["gap"] = new_df.groupby("Unit")["date"].diff().dt.days

    gap_rows = new_df[new_df["gap"] > 1]
    gap_counts = gap_rows[["date", "gap"]].value_counts()

    missing_days = []
    for date, gap in gap_counts.index:
        missing_days.extend(pd.date_range(end=date - pd.Timedelta(days=1), periods=int(gap) - 1))
    missing_days = pd.DatetimeIndex(sorted(missing_days))

    print(f"{len(missing_days)} days have no daily report:", ", ".join(missing_days.strftime("%Y-%m-%d")))
    yearly_data = pd.concat([ pd.read_csv(yearly_dir/f'{year}PowerStatus.txt', sep="|", parse_dates=["ReportDt"]) for year in missing_days.year.unique() ])

    missing_data = yearly_data[ yearly_data["ReportDt"].isin(missing_days) ].sort_values("ReportDt").reset_index(drop=True)

    missing_data["Unit"] = missing_data["Unit"].replace(UNIT_RENAMES)

    missing_data = missing_data[missing_data["Unit"].isin(new_df["Unit"].unique())].reset_index(drop=True)

    missing_data = missing_data.rename(columns={"ReportDt":"date"})
    missing_data["Down"] = np.nan # Matching new_df convention
    missing_data["Reason"] = ["Not Available" if p < 100 else np.nan for p in missing_data["Power"]]

    # For the region information we need a lookup table between Unit --> Region

    unit_region_map = new_df.groupby("Unit")["Region"].first()

    missing_data["Region"] = missing_data["Unit"].map(unit_region_map)

    new_df = pd.concat([new_df, missing_data], ignore_index=True)
    new_df = new_df.sort_values(["Unit", "date"]).reset_index(drop=True)

    columns_to_keep = ["date", "Region", "Unit", "Power", "Down", "Reason"]

    for year, year_df in new_df.groupby(new_df["date"].dt.year):
        year_df[columns_to_keep].to_csv(cleaned_dir/f"CleanedData_{year}.csv", sep="\t", index=False)

    print("After removing currently unoperating reactors, we now have", new_df["Unit"].nunique(), "reactors")

    return new_df

def plot_power_output(df):

    fig, axes = plt.subplots(5,4,figsize=(15,15))
    axes = axes.flatten()

    region1_reactors = df[df["Region"] == "Region 1"]
    region1_reactor_names = sorted(region1_reactors["Unit"].unique())

    for i in range(len(region1_reactor_names)):
        axes[i].scatter(region1_reactors[ region1_reactors["Unit"] == region1_reactor_names[i] ].date, region1_reactors[ region1_reactors["Unit"] == region1_reactor_names[i] ].Power, s=2)
        axes[i].set_xlabel("Year")
        axes[i].set_ylabel("Power Output")
        axes[i].set_title(f'{region1_reactor_names[i]}')

    plt.tight_layout()
    plt.show()

def outages(df):

    rows = []

    for unit, unit_df in df.groupby("Unit"):

        region = unit_df["Region"].iloc[0]
        count = 0

        for power, date, reason in zip(unit_df["Power"], unit_df["date"], unit_df["Reason"]):

            if power == 0:
                if count == 0:
                    start = date
                    outage_reason = np.nan # Initialize to missing
                    refuel_days = 0 # Initialize to 0
                if pd.isna(outage_reason) and pd.notna(reason):
                    outage_reason = reason
                if pd.notna(reason) and ("refuel" in reason.lower() or "defuel" in reason.lower()):
                    refuel_days += 1
                count += 1
                end = date
            elif count > 0:
                rows.append({ "Unit":unit, "Region": region, "Start_Date": start, "End_Date": end, "Outage_Duration": count, "Reason":outage_reason, "Refuel_Days": refuel_days})
                count = 0

        if count > 0:
            rows.append({ "Unit":unit, "Region": region, "Start_Date": start, "End_Date": end, "Outage_Duration": count , "Reason": outage_reason, "Refuel_Days": refuel_days})

    return pd.DataFrame(rows)

def plot_outages(df):

    region = "Region 1"

    region_outages = df[df["Region"] == region]
    region_reactors = region_outages["Unit"].unique()

    n_plots = len(region_reactors)
    n_cols = 4
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10,10))
    axes = axes.flatten()

    for i in range(len(region_reactors)):

        axes[i].hist(region_outages[ region_outages["Unit"] == region_reactors[i] ].Outage_Duration, bins=10)
        axes[i].set_xlabel("outage duration")
        axes[i].set_ylabel("Frequency")
        axes[i].set_title(f'{region_reactors[i]}')

    plt.tight_layout()
    plt.show()

def clean_reasons(outages):

    n_before = outages["Reason"].nunique()

    # "REFUELING OUTAGE" and "Refueling outage" are listed as two different reasons, 
    # so we make everything lowercase .str only works on Pandas Series

    outages["Reason"] = outages["Reason"].str.lower()
    outages["Reason"] = outages["Reason"].str.strip()

    # Replaces one or more whitespace characters in a row to a single space
    # some of the reasons fields have ("en \t57285" --> "en 57285")
    # Another example is "forced outage\r\nen 54508" --> "forced outage en 54508"
    # without regex=True, pandas is literally going to look for \, s and + rather than a pattern

    outages["Reason"] = outages["Reason"].str.replace(r"\s+", " ", regex=True)

    print("\n")
    print("There are", n_before, "different reasons listed for the output power going down to 0.")
    print("Many of them are the same reason written differently, e.g. 'REFUELING OUTAGE' and 'Refueling outage'.")
    print("After making them lowercase and fixing the spacing there are", outages["Reason"].nunique(), "left,")
    print("most of which are different wordings of a few causes, so next we group them into categories.")
    print("\n")

    return outages

def categorize_outages(outages):

    # Creating a new column for the category division
    # We will put each outage into one of: refueling, planned/maintenance, unplanned, weather related or unknown
    outages["outage_reason"] = outages["Reason"]

    # Any kinds of trips or scram or leaks account for unplanned outages, and so do equipment issues, problems and failures
    # sometimes there are planned trips as well. Those would be overwritten to planned downstream

    mask = outages["Reason"].str.contains("trip|leak|scram|issue|risk|problem|failure|turbine", na=False)
    outages.loc[ mask, "outage_reason"] = "unplanned"

    # A leak or problem repaired during a maintenance outage is still a maintenance outage
    # "maint" also catches the spellings "maint." and "maintainance"

    outages.loc[ outages["Reason"].str.contains("maint", na=False), "outage_reason"] = "maintenance outage"

    # A trip while shutting down for refueling is still a refueling outage, and so is a refueling outage with maintenance as a side note
    # e.g. "scheduled refueling outage the uat will be oos for maintenance"

    outages.loc[ outages["Reason"].str.contains("refuel|defuel|fuel load|fueling outage", na=False), "outage_reason"] = "refueling outage"

    # There are reasons like "forced maintenance outage", "emergent maintenance outage – stator cooling leak in generator", 
    # "shutdown for unplanned maintenance" Even though it has the word maintenace the primary reason is an unplanned outage, so the order here MATTERS

    mask = outages["Reason"].str.contains("forced|unplanned|unscheduled|emergent", na=False)
    outages.loc[ mask, "outage_reason"] = "unplanned"

    # Tech spec shutdowns are unplanned. Operators write them in several ways: "tech spec", "technical specification", "ts required", "ts-required", "ts reqd", "ts shutdown", "lco required"

    outages.loc[ outages["Reason"].str.contains("tech spec|technical spec|ts required|ts-required|ts reqd|ts shutdown|lco required", na=False), "outage_reason"] = "unplanned"

    # Weather related outages. "cold weather alert" or "high wind warning" are not used here as they are grid operator notices

    mask = outages["Reason"].str.contains("hurricane|adverse weather|electrical storm|lightning|lighting strike|icing|earthquake", na=False)
    outages.loc[ mask, "outage_reason"] = "weather related outage"

    # From here on we only label rows that have not been categorized yet, so that we don't overwrite the labels above.
    # uncategorized() is recomputed every time it is called, so it always reflects the latest labels.

    labels = ["refueling outage", "maintenance outage", "unplanned", "weather related outage", "planned outage", "unknown", "other"]

    def uncategorized():
        return ~outages["outage_reason"].isin(labels)

    # "see en #" refers to the event reports filed with the NRC (10 CFR 50.72), so these are most likely unplanned outages
    mask = (outages["Reason"].str.contains("see en|see event|en#|en #", na=False) | outages["Reason"].str.contains("en5", na=False) | outages["Reason"].str.contains("en 5", na=False) | outages["Reason"].str.contains("en4", na=False))
    outages.loc[ mask & uncategorized(), "outage_reason"] = "unplanned"

    # Planned but not refueling. "planned" is also part of "unplanned", which is why this only applies to the uncategorized rows
    mask = outages["Reason"].str.contains("planned|scheduled|mid cycle|mid-cycle|midcycle", na=False)
    outages.loc[ mask & uncategorized(), "outage_reason"] = "planned outage"

    # Equipment failures, chemistry problems and things clogging the cooling water intake
    mask = outages["Reason"].str.contains("fail|malfunction|lockout|lifted|loss of|loop|dropped rod|misalign|alignment|chemistry|sodium|salp|grassing|intake|fire|required|event|lco", na=False)
    outages.loc[ mask & uncategorized(), "outage_reason"] = "unplanned"

    # Shutdowns to repair, replace or inspect equipment
    mask = outages["Reason"].str.contains("repair|replac|troubleshoot|inspect|seal|valve|pump|generator|transformer|feedwater|heater|rcp|recirc|srv|msiv|ehc|condenser|rod|instrument|detector|work|concern|balance|modification|exciter|cooler|xfmr|mcc|bearing|oil level", na=False)
    outages.loc[ mask & uncategorized(), "outage_reason"] = "maintenance outage"

    refuel_share = outages["Refuel_Days"] / outages["Outage_Duration"]
    mask = (refuel_share >= 0.5)
    outages.loc[ mask, "outage_reason" ] = "refueling outage"

    # Everything that is left
    # Startup / testing notes: these describe the restart, not the cause. Missing reasons and the "not available" gap days are unknown too.
    # The ambiguous cold weather alert (with nothing else) is also categorized as unknown

    outages.loc[ uncategorized(), "outage_reason"] = "unknown"

    outages["outage_reason"] = outages["outage_reason"].replace({ "planned outage": "planned/maintenance outage", "maintenance outage":"planned/maintenance outage" })

    print("After this cleanup we have", outages["outage_reason"].nunique(), "unique reasons for outages")

    return outages

def outage_summary(outages):

    # count only counts the number of rows but sum would sum up the numbers in the outage_duration
    summary = outages.groupby("outage_reason")["Outage_Duration"].agg(["count", "median", "sum"]).sort_values("count", ascending=False)
    summary["percent_of_outage_days"] = (100 * summary["sum"] / summary["sum"].sum()).round(1)

    print(summary)

    return summary

def plot_outage_categories(outages, x_limits):

    categories = outages["outage_reason"].value_counts().index

    n_cols = 5
    n_rows = math.ceil(len(categories) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.5 * n_rows))
    axes = axes.flatten()

    for i in range(len(categories)):

        limit = x_limits[categories[i]]
        axes[i].hist(outages[ outages["outage_reason"] == categories[i] ].Outage_Duration, bins=30, range=(0, limit))
        axes[i].set_xlabel("outage duration (days)")
        axes[i].set_ylabel("Frequency")
        axes[i].set_title(f'{categories[i]}')

    # Hide the empty plots
    for j in range(len(categories), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def main():

    daily_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    df = generate_df()

    print(" ")
    print("From the daily reports spanning 2000-2025 we have", df["Unit"].nunique(), "reactors.")
    print("But not all of those are currently operating")
    print(" ")

    operating_df = initial_cleanup(df)
    plot_power_output(operating_df)

    outages_df = outages(operating_df)
    plot_outages(outages_df)

    outages_df = clean_reasons(outages_df)
    outages_df = categorize_outages(outages_df)

    outage_summary(outages_df)
    plot_outage_categories(outages_df, { "refueling outage":100, "unplanned":25, "planned/maintenance outage":30,  "unknown": 20, "weather related outage": 100})

    # PreliminaryForecasts and AdvancedForecasts read this file (tab separated)
    outages_df[["Unit", "Region", "Start_Date", "End_Date", "Outage_Duration", "outage_reason"]].to_csv(cleaned_dir/"outages_data.csv", sep="\t", index=False)

if __name__ == "__main__":
    main()

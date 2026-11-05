#!/usr/bin/env python
import re
from pathlib import Path
from urllib.parse import urljoin

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = "https://www.nrc.gov/facilities-safety/facility-finder/reactors"
STATES_SHAPEFILE = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip"
# Paths are relative to the repo, so the code runs from any clone
OUTDIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "nrc"

REACTOR_LINK = re.compile(r"/(?:facility-finder/reactors|combined-license-holders-for-new-reactors)/([a-z0-9-]+)/?$")


def download_page(url, html_file):
    
    if html_file.exists():
        return
    # requests.get() sends the HTTP request and gets back the HTML
    webpage = requests.get(url, timeout=60)
    # Stops the script with an error if the page could not be fetched
    webpage.raise_for_status()
    html_file.write_bytes(webpage.content)


def parse_reactors(html_file):

    # BeautifulSoup reads the HTML so we can search it
    soup = BeautifulSoup(html_file.read_text(encoding="utf-8"), "html.parser")

    heading = soup.find("h2", id="listState")
    table = heading.find_next("table")

    # Headings for the Columns (Region1, 2, 3 and 4) Contained in the <th> objects
    all_regions = [t.get_text(strip=True) for t in table.find_all("th")]

    # The 4 columns can be access by finding all the <td> objects
    columns = table.find_all("td")

    region_data = []

    for region, column in zip(all_regions, columns):

        # Within each column there are multiple states which can be accessed via <dl> blocks.
        for state_block in column.find_all("dl"):
            state_name = state_block.find("dt").get_text(strip=True)  # Names in the <dt> blocks
            # Links in the <dd> blocks, there can be multiple reactors in each state so we are using "find_all"
            for reactor_block in state_block.find("dd").find_all("a", href=True):
                full_url = urljoin(URL, reactor_block["href"])
                match = REACTOR_LINK.search(full_url)
                name = reactor_block.get_text("", strip=True)
                if match:
                    region_data.append({"code": match.group(1), "name": name, "Region": region, "State": state_name})

    return pd.DataFrame(region_data).sort_values("name").reset_index(drop=True)


def plot_region_map(df):

    # The other columns from the data frame will be dropped. Reset_Index will create a DataFrame with the number of reactors column as reactors. Without reset_index, the groupby() will return a pandas Series.

    counts = df.groupby(["Region", "State"]).size().reset_index(name="reactors")
    print(counts)

    # This is a table containing the state names and their corresponding geometries
    states = gpd.read_file(STATES_SHAPEFILE)

    # Not including Alaska, Hawaii and Puerto Rico
    states = states.query("NAME not in ['Alaska', 'Hawaii', 'Puerto Rico']")

    # A more usual way of plotting. Without this the map would appear flat.
    states = states.to_crs(epsg=5070)

    # We merge the two tables by the States (the corresponding column is called Name in the states table and State in the counts table)
    # how = left means whichever rows are not matched they are kept with NaN in the reactors column
    states = states.merge(counts, left_on="NAME", right_on="State", how="left")

    # Coloring by Region
    ax = states.plot(column="Region", legend=True, edgecolor="white", figsize=(12, 8), missing_kwds={"color": "lightgrey", "label": "No reactors"}, legend_kwds={"loc": "lower left"})

    # Only writing the number of reactors for the states with nuclear reactors. The rows with NaN in the reactors column are dropped
    labeled = states.dropna(subset=["reactors"])
    for point, n in zip(labeled.representative_point(), labeled["reactors"]):
        ax.annotate(int(n), (point.x, point.y))

    ax.set_title(f"{len(df)} operating US nuclear reactors by NRC region")
    ax.set_axis_off()


def plot_region_bar(df):
    
    fig, ax = plt.subplots()
    df["Region"].value_counts().sort_index().plot.bar(ax=ax)
    ax.set_ylabel("Number of reactors")


def main():

    OUTDIR.mkdir(parents=True, exist_ok=True)
    html_file = OUTDIR / "reactor_list.html"

    download_page(URL, html_file)
    df = parse_reactors(html_file)
    df.to_csv(OUTDIR / "reactor_list.csv", index=False)

    plot_region_map(df)
    plot_region_bar(df)

    plt.show()


if __name__ == "__main__":
    main()

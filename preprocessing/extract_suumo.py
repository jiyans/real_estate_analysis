#!/usr/bin/env python3
from pathlib import Path
import os
import re
import pandas as pd

SITE_ROOT =  "https://suumo.jp/"


def flatten_series(ser):
    assert ser.apply(lambda x: isinstance(x, list)).all(), "Elements of ser arent lsits"
    assert ser.apply(lambda x: len(x)==1).all(), "Not all elements are of length 1)"
    return ser.apply(lambda x: x[0])

def clean_apt_floor_strings(s):
    s = s.split("-")[0]
    s = "-".join(s.split("B"))
    if s == "":
        s = "1"
    return s

methodreg = re.compile(r"(\D*)(\d+)(\D+)")

def get_closest(station_dict):
    station_dict
    s_dict = sorted(station_dict.items(), key=lambda x: int(methodreg.search(x[1])[2]))
    station, dist_string = s_dict[0]
    matches =  methodreg.search(dist_string)
    method = matches[1]
    time = int(matches[2])
    unit = matches[3]

    return station, method, time, unit

def extract_suumo(df, site_root=SITE_ROOT):
    # Get rid of rows without images
    df = df[~df["images"].apply(lambda x: x == [])].copy()
    assert df["apt_size"].str.endswith("m").all(), "Not all apartmentsizes end with m"
    df["apt_size"] = df["apt_size"].str.slice(0, -1).astype(float)

    #Extract number of floors above ground
    df["b_no_floors"] = df["b_no_floors"].str.extract(r"(\d+)階建")[0].astype(float)

    assert df["apt_rent"].str.endswith("万円").all(), "Not all rents in 万円"
    df["apt_rent"] = df["apt_rent"].str.slice(0, -2).astype(float)



    df["b_age"] = flatten_series(df["b_age"])
    assert (pd.Series(df["b_age"].unique()).apply(len) == 2).sum() == 1, "There should be only 1 element with length 2, and that is `新築`"
    df["b_age"] = df["b_age"].str.slice(1,-1).apply(lambda x: "0" if x == "" else x).astype(float)

    assert df["apt_admin_price"].str.replace("-", "0円").str.endswith("円").all(), "All prices should either be 0 or end with 円"
    df["apt_admin_price"] = df["apt_admin_price"].str.replace("-", "0円").str.slice(0, -1).astype(float)

    # Extract the floor number,
    df["apt_floor"] = flatten_series(df["apt_floor"])
    df["apt_floor"] = df["apt_floor"].apply(clean_apt_floor_strings).str.extract(r"(-?\d+)")[0]

    # Put full links in there
    df["full_apt_detail_link"] = df["apt_detail_link"].apply(lambda x: site_root+x)

    # Doing these as well now
    assert df["apt_thanks_fee"].replace("-", "0万円").str.endswith("万円").all(), "All prices should have same unit"
    df["apt_thanks_fee"] = df["apt_thanks_fee"].replace("-", "0万円").str.slice(0, -2)

    assert df["apt_deposit"].replace("-", "0万円").str.endswith("万円").all(), "All prices should have same unit"
    df["apt_deposit"] = df["apt_deposit"].replace("-", "0万円").str.slice(0, -2)

    df["rel_image_paths"] = df["images"].apply(lambda x: x[0]["path"])

    df[["station", "method", "time_to_station", "unit"]] =  df[["b_closest_stations"]].apply(
        lambda x: get_closest(x["b_closest_stations"]), axis=1, result_type="expand")

    df["apt_floor"] = df["apt_floor"].astype(float)
    df["apt_thanks_fee"] = df["apt_thanks_fee"].astype(float)
    df["apt_deposit"] = df["apt_deposit"].astype(float)
    df = df.drop(["b_closest_stations", "image_urls", "images"], axis=1).drop_duplicates().reset_index(drop=True)

    return df

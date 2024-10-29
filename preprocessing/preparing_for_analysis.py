#!/usr/bin/env python3
import os
import hashlib
import re
from pathlib import Path

import pandas as pd
from fastai.tabular.all import df_shrink

basepath = Path("data/")
rawpath = basepath / "tokyo_2024_10_29"
fname = rawpath / "tokyo_all.jsonlines"

# %%
df = pd.read_json(
    fname,
    lines=True,
)
assert isinstance(df, pd.DataFrame)
print(df.columns)
# %%

eventual_dtypes = {
    "b_name": "str",
    "apt_detail_link": "str",
    "b_address": "str",
    "b_age_int": "int",
    "b_closest_station_1": "category",
    "b_distance_station_1": "float",
    "b_method_station_1": "category",
    "b_closest_station_2": "category",
    "b_distance_station_2": "float",
    "b_method_station_2": "category",
    "b_closest_station_3": "category",
    "b_distance_station_3": "float",
    "b_method_station_3": "category",
    "b_no_floors_int": "int",
    "apt_size_num": "float",
    "apt_rent_num": "float",
    "apt_style": "category",
    # TODO This might work better as a category
    "apt_floor_num": "int",
    "apt_admin_price_num": "float",
    "apt_thanks_fee_num": "float",
    "apt_total_rent_num": "float",
    "apt_deposit_num": "float",
    "images_url": "str",
    "images_path": "str",
}


# %%
def get_closest_station_parts(t):
    if isinstance(t, float):
        return pd.Series([None, None, None])
    if t is None:
        return pd.Series([None, None, None])
    station = t[0]
    s = re.search(r"(歩|バス|車)(\d+)分", t[1])
    if s:
        method, time = s[1], s[2]
    else:
        method, time = None, None
        print(f"Failed to parse {t[1]}")
    return pd.Series([station, method, time])

def get_sha1_hash(text):
    # Convert string to bytes if it isn't already
    if isinstance(text, str):
        text = text.encode('utf-8')
    return hashlib.sha1(text).hexdigest()

def prep_data(df):
    print("preparing easy columns")
    df["b_age"] = df["b_age"].apply(lambda x: x[0]).str.replace("新築", "0年")
    df["b_age_int"] = df["b_age"].str.extract(r"(\d+)年")[0].astype(int)
    print("Age done ")
    df["b_no_floors"] = df["b_no_floors"].str.replace("平屋", "1階建")
    df["b_no_floors_int"] = df["b_no_floors"].str.extract(r"(\d+)階建")[0].astype(int)
    print("building floors done")

    assert df["apt_size"].str.endswith("m").all()
    df["apt_size_num"] = df["apt_size"].str.slice(0, -1).astype(float)
    print("Apt size done ")

    assert df["apt_rent"].str.endswith("万円").all()
    df["apt_rent_num"] = df["apt_rent"].str.slice(0, -2).astype(float)

    print("apt rent num done")

    df["apt_floor_num"] = df["apt_floor"].replace("-", "1階").str.extract(r"(\d+)階")[0].fillna(1).astype(int)
    print("apt floors done")

    df["apt_admin_price"] = df["apt_admin_price"].replace("-", "0円")
    assert df["apt_admin_price"].str.endswith("円").all()
    df["apt_admin_price_num"] = df["apt_admin_price"].str.slice(0, -1).astype(float)

    print("Admin price done ")
    df["apt_thanks_fee"] = df["apt_thanks_fee"].replace("-", "0万円")
    assert df["apt_thanks_fee"].str.endswith("万円").all()
    df["apt_thanks_fee_num"] = (
        df["apt_thanks_fee"].str.slice(0, -2).astype(float) * 10000
    )

    print("Thanks fee done ")
    df["apt_deposit"] = df["apt_deposit"].replace("-", "0万円")
    assert df["apt_deposit"].str.endswith("万円").all()
    df["apt_deposit_num"] = df["apt_deposit"].str.slice(0, -2).astype(float) * 10000
    print("Deposit done ")

    df["apt_total_rent_num"] = df["apt_rent_num"] + df["apt_admin_price_num"]
    df["images_url"] = df["image_urls"].apply(lambda x: x[0])
    df["images_path"] = df["images_url"].apply(lambda x: get_sha1_hash(x) + ".jpg")

    print("Images done ")

    df[["b_closest_1", "b_closest_2", "b_closest_3"]] = df["b_closest_stations"].apply(
        lambda x: pd.Series(x.items())
    )
    print("splitting closest stations")
    for i in range(1, 4):

        df[
            [
                f"b_closest_station_{i}",
                f"b_method_station_{i}",
                f"b_distance_station_{i}",
            ]
        ] = df[f"b_closest_{i}"].apply(get_closest_station_parts)
        print(f"Done with {i}th part")
    df["b_distance_station_1"] = df["b_distance_station_1"].astype(float)
    df["b_distance_station_2"] = df["b_distance_station_2"].astype(float)
    df["b_distance_station_3"] = df["b_distance_station_3"].astype(float)
    print("Distances done ")
    return df


# %%
df_p = prep_data(df)
# %%
if not os.path.exists(basepath / "processed"):
    os.makedirs(basepath / "processed")

# %%
df_slim = df_p[eventual_dtypes.keys()].copy()
df_slim = df_shrink(df_slim)

df_slim.to_parquet(rawpath /  "slim.parquet")

# %%

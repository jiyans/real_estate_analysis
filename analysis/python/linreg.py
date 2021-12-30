#!/usr/bin/env python3
import matplotlib.font_manager as fm

font_dirs = ["/Users/jiyanschneider/Library/Fonts/"]
font_files = fm.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    fm.fontManager.addfont(font_file)

import matplotlib.pyplot as plt

plt.style.use(["science", "nature", "notebook"])
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer", "Noto Serif CJK jp"],
        "font.size": 11,
        "text.usetex": False,
        "mathtext.fontset": "stixsans",
        "figure.figsize": (5, 3),
    }
)

from prep import *
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv("../data/viz_learn_preds.csv", index_col=0)


def linear_model(formula, df, targ, model_name=None):
    results = smf.ols(formula.format(targ), data=df).fit()
    preds = results.predict(df)
    targs = df[targ]
    if model_name is None:
        model_name = formula.format(targ)
    fig, annot = plot_regression_once(preds, targs, model_name)
    return results, fig


def lm(formula, df, targ, model_name=None):
    results = smf.ols(formula.format(targ), data=df).fit()
    return results


def set_base(ser, base):
    cats = list(ser.unique())
    idx = cats.index(base)
    cats[idx] = cats[0]
    cats[0] = base
    return pd.Categorical(ser, categories=cats)


df["station"] = set_base(df["station"], "ＪＲ山手線/浜松町駅")
df["method"] = set_base(df["method"], "歩")
df["apt_style"] = df["apt_style"].apply(apt_style_clean)
df["time_to_station_sq"] = df["time_to_station"] ** 2
df["apt_style"] = set_base(df["apt_style"], "ワンルーム")

mod = lm(
    "{} ~ b_age"
    "+ apt_size"
    "+ b_no_floors"
    "+ apt_floor"
    "+ apt_admin_price"
    "+ time_to_station"
    "+ time_to_station_sq"
    "+ C(method)"
    "+ C(station)"
    "+ apt_style"
    "+ viz_preds",
    df,
    "log_apt_rent",
    model_name="Simple",
)

from stargazer.stargazer import Stargazer
sg = Stargazer([mod] )

sg.covariate_order(["apt_floor", "b_no_floors"])

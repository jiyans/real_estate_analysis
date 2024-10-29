#!/usr/bin/env python3
# %%
import pandas as pd
import os

suumofile = (
    "https://keio.box.com/shared/static/l8chhni4daldzc876tjql6s1zo8md90r.jsonlines"
)

# %%
if not os.path.exists("suumo.pkl"):
    df = pd.read_json(suumofile, lines=True)
    df.to_pickle("suumo.pkl")
else:
    df = pd.read_pickle("suumo.pkl")

# %%

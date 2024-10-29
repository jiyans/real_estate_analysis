#!/usr/bin/env python
# coding: utf-8

# # Imports and Setup

# In[56]:


get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "1")
get_ipython().run_line_magic("aimport", "prep")
get_ipython().run_line_magic("config", 'InlineBackend.figure_formats = ["jpg"]')

# In[43]:

# k
import matplotlib.font_manager as fm

font_dirs = ["/Users/jiyanschneider/Library/Fonts/"]
font_files = fm.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    fm.fontManager.addfont(font_file)

import matplotlib.pyplot as plt

plt.style.use(["science", "nature", "notebook"])
plt.rcParams.update(
    {
        "font.family": "serif",  # specify font family here
        "font.serif": ["Computer", "Noto Serif CJK jp"],  # specify font here
        "font.size": 11,
        "text.usetex": False,
        "mathtext.fontset": "stixsans",
        "figure.figsize": (5, 3),
    }
)  # specify font size here

# plt.rcParams['figure.dpi'] = 100
# plt.rcParams['savefig.dpi'] = 300
# plt.rcParams['font.size'] = 20
# plt.rcParams['legend.fontsize'] = 20
# plt.rcParams['figure.titlesize'] = 24

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from prep import *

x = 1
# In[44]:


df = pd.read_csv("../data/suumo.csv", low_memory=False, index_col=0)
df["apt_style"] = df["apt_style"].apply(apt_style_clean)

df["log_apt_rent"] = np.log(df["apt_rent"])


# ##  Fit regression model
#
# Also declare two helper functions

# In[48]:


def linear_model(formula, df, targ, model_name=None):
    results = smf.ols(formula.format(targ), data=df).fit()
    preds = results.predict(df)
    targs = df[targ]
    if model_name is None:
        model_name = formula.format(targ)
    fig, annot = plot_regression_once(preds, targs, model_name)
    print(annot)
    plt.savefig("../../paper/assets/" + model_name + ".jpg")
    return results, fig


def set_base(ser, base):
    cats = list(ser.unique())
    idx = cats.index(base)
    cats[idx] = cats[0]
    cats[0] = base
    return pd.Categorical(ser, categories=cats)


# In[49]:


df["station"] = set_base(df["station"], "ＪＲ山手線/浜松町駅")
df["apt_style"] = set_base(df["apt_style"], "ワンルーム")
df["method"] = set_base(df["method"], "歩")

# ### Simple Regression of easiest variables on `rent`

# In[137]:


res, p = linear_model(
    "{} ~"
    "+ b_no_floors "
    "+ apt_size"
    "+ apt_admin_price"
    "+ apt_floor"
    "+ method * time_to_station",
    df,
    "apt_rent",
    model_name="Linear Regression",
)
res.summary()

### Simple regression `log_rent` on simple variables
# In[139]:

res, p = linear_model(
    "{} ~" "+ b_no_floors " "+ apt_size"
    # "+ apt_style"
    "+ apt_admin_price" "+ apt_floor" "+ method * time_to_station" "+ time_to_station^2"
    # "+ station"
    ,
    df,
    "log_apt_rent",
    model_name="Linear Regression log",
)
res.summary()


# ### Regression on rent with `apt_style`

# In[131]:


res, p = linear_model(
    "{} ~ b_no_floors "
    "+ apt_size"
    "+ apt_style"
    "+ apt_admin_price"
    "+ apt_floor"
    "+ method * time_to_station"
    # "+ station"
    ,
    df,
    "apt_rent",
    model_name="style and apt_rent",
)
res.summary()


# ### Regression on log rent with style

# In[160]:


res, p = linear_model(
    "{} ~ b_no_floors "
    "+ apt_size"
    "+ apt_style"
    "+ apt_admin_price"
    "+ apt_floor"
    "+ method * time_to_station"
    # "+ station"
    ,
    df,
    "log_apt_rent",
    model_name="With apt_style, log rent",
)
res.summary()


# ### Regression on apt_rent using all variables

# In[132]:


m3, p = linear_model(
    "{} ~ b_no_floors "
    "+ apt_size"
    "+ apt_style"
    "+ apt_admin_price"
    "+ apt_floor"
    "+ method * time_to_station"
    "+ station",
    df,
    "apt_rent",
    model_name="Style, station, rent",
)
m3.summary()


# ### Regerssion on all variables using `log_rent`

# In[191]:


m4, p = linear_model(
    "{} ~ b_no_floors "
    "+ apt_size"
    "+ apt_style"
    "+ apt_admin_price"
    "+ apt_floor"
    "+ method * time_to_station"
    "+ method * np.sqrt(time_to_station)"
    "+ station",
    df,
    "log_apt_rent",
    model_name="Station, Style, log rent",
)


# We can see that it is mostly overvaluing the most places

# In[131]:


from statsmodels.iolib.summary2 import summary_col

# In[132]:


summary_col([m3, m4])


# In[62]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# In[63]:


df.groupby("station").count()["b_name"].sort_values().iloc[-50:]


# In[64]:


df.groupby("station").count()["b_name"].sort_values().iloc[:50]


# In[92]:


cont = [
    "b_age",
    "b_no_floors",
    "apt_size",
    "apt_admin_price",
    "apt_floor",
    "time_to_station",
]

cat = [
    "method",
    "apt_style",
    "station",
]


# In[108]:


cols = [
    "b_age",
    "b_no_floors",
    "apt_size",
    "apt_style",
    "station",
    "apt_admin_price",
    "apt_floor",
    "method",
    "time_to_station",
]
targs = "log_apt_rent"
dm = pd.get_dummies(df[cols], drop_first=True)
rf = RandomForestRegressor(oob_score=True)


# In[ ]:


df.describe()


# In[113]:


df.iloc[df["apt_admin_price"].argmax()]["full_apt_detail_link"]


# In[127]:


df.sort_values("apt_admin_price", ascending=False).iloc[:10][
    "full_apt_detail_link"
].iloc[2]


# In[123]:


df[cols + [targs]].describe().T.drop("count", axis=1)  # .to_latex(float_format="%.2f")


# In[ ]:


print(
    df[cols + [targs]].describe(include=["category"]).T.drop("count", axis=1).to_latex()
)


# In[107]:


X, y = dm, df[targs]
tr_X, te_X, tr_y, te_y = train_test_split(dm, df[targs], random_state=42)


# In[ ]:


train_preds = rf.predict(tr_X)
test_preds = rf.predict(te_X)


# In[94]:


import numpy as np
from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(oob_score=True)

# 10-Fold Cross validation
# np.mean(cross_val_score(clf, tr_X, tr_y, cv=10))

# In[95]:


get_ipython().run_cell_magic("time", "", "clf.fit(tr_X, tr_y)")

# In[134]:


clf.oob_score_


# In[84]:


get_ipython().run_cell_magic("time", "", "cross_val_score(clf, tr_X, tr_y, cv=5)")


# In[81]:


train_preds = clf.predict(tr_X)
test_preds = clf.predict(te_X)


# In[128]:


fig, annot = plot_regression(train_preds, tr_y, test_preds, te_y, "Random Forest")


# In[198]:


print(annot)


# In[130]:


plt.savefig("../../paper/assets/Random Forest.jpg")


# In[139]:


((df["log_apt_rent"] - df["log_apt_rent"].mean()) ** 2).mean()

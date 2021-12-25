
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from collections import namedtuple
import numpy as np

Stats=namedtuple("Stats", ["corr", "mse", "rmse", "N"])
def apt_style_clean(s):
    fd,sd = s[0],s[1]
    if fd.isnumeric():
        if int(fd)>4 or sd.isnumeric(): return "5以上"
        else: return s
    else: return s

def annotate_string(sl):
    r1 = '{:7s}{:15s}{:15s}{:15s}{:5s}'.format("", "Correlation", "MSE", "RMSE", "N")
    r2 = str("-"*57)
    row_strings=[r1, r2]
    for i in sl:
        r3 = ('{:<7s}{:<15.3f}{:<15.2f}{:<15.2f}{:5d} '.format("Train", i.corr,i.mse,i.rmse, i.N))
        row_strings.append(r3)
    annotated_string = "\n".join(row_strings)
    return annotated_string

def plot_regression_once(preds,targs, model_name):
    targs = pd.Series(targs).reset_index(drop=True)
    preds - pd.Series(preds).reset_index(drop=True)
    m_se = mean_squared_error(preds, targs)
    trainstats = Stats(
        mse = m_se,
        rmse = np.sqrt(m_se),
        N = preds.shape[0],
        corr = preds.corr(targs),
    )
    plt.figure(figsize=(16, 9))
    plt.scatter(x=targs, y=preds, alpha=0.10)
    m = max(targs.max(), preds.max())
    corrcords = (0.0, round(m*0.8))
    
    plt.plot([0, m], [0, m], label="y = x", c="purple")
    plt.legend(fontsize=15)
    plt.ylabel("Predictions", fontsize=18)
    plt.xlabel("Ground Truth", fontsize=18)
    annotation_string=annotate_string([trainstats])
    plt.annotate(annotation_string,
                 corrcords, 
                fontsize=15, family="monospace")
    return plt.title(f"{model_name}" , fontsize=22)

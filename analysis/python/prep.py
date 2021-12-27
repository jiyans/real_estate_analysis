import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from collections import namedtuple
import numpy as np

Stats = namedtuple("Stats", ["corr", "mse", "rmse", "N"])


def apt_style_clean(s):
    fd, sd = s[0], s[1]
    if fd.isnumeric():
        if int(fd) > 4 or sd.isnumeric():
            return "5以上"
        else:
            return s
    else:
        return s


def annotate_string(sl):
    r1 = "{:7s}{:15s}{:15s}{:15s}{:5s}".format("", "Correlation", "MSE", "RMSE", "N")
    r2 = str("-" * 57)
    row_strings = [r1, r2]
    for t, i in zip(["Train", "Test"], sl):
        r3 = "{:<7s}{:<15.3f}{:<15.2f}{:<15.2f}{:5d} ".format(
            t, i.corr, i.mse, i.rmse, i.N
        )
        row_strings.append(r3)
    annotated_string = "\n".join(row_strings)
    return annotated_string


def plot_regression_once(preds, targs, model_name):
    targs = pd.Series(targs).reset_index(drop=True)
    preds - pd.Series(preds).reset_index(drop=True)
    m_se = mean_squared_error(preds, targs)
    trainstats = Stats(
        mse=m_se,
        rmse=np.sqrt(m_se),
        N=preds.shape[0],
        corr=preds.corr(targs),
    )
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    plt.scatter(x=targs, y=preds, alpha=0.10)
    plt.scatter(x=targs, y=preds, alpha=0.10)
    m = max(targs.max(), preds.max())

    plt.plot([0, m], [0, m], label="y = x", alpha=0.6)
    plt.legend(fontsize=15)
    plt.ylabel("Predictions", fontsize=18)
    plt.xlabel("Ground Truth", fontsize=18)
    annotation_string = annotate_string([trainstats])
    corrcords = (-3.5, -6)
    # corrcords = (0.0, round(m*0.8))
    plt.annotate(
        annotation_string,
        corrcords,
        xycoords="data",
        fontsize=12,
        family="monospace",
        annotation_clip=False,
    )
    plt.title(f"{model_name}", fontsize=22)
    return fig, annotation_string


def plot_regression(tp, tt, vp, vt, model_name=""):
    # Preds steals the indices, so we remove them from targs as
    tt = pd.Series(tt).reset_index(drop=True)
    tp = pd.Series(tp).reset_index(drop=True)

    vt = pd.Series(vt).reset_index(drop=True)
    vp = pd.Series(vp).reset_index(drop=True)

    mse_test = mean_squared_error(vp, vt)
    mse_train = mean_squared_error(tp, tt)
    N_test = vp.shape[0]
    N_train = tp.shape[0]
    corr_test = vp.corr(vt)
    corr_train = tp.corr(tt)

    test_stats = Stats(corr=corr_test, mse=mse_test, rmse=np.sqrt(mse_test), N=N_test)
    train_stats = Stats(
        corr=corr_train, mse=mse_train, rmse=np.sqrt(mse_train), N=N_train
    )

    fig = plt.figure(figsize=(5, 4))
    plt.scatter(x=vt, y=vp, alpha=0.10, label="Test")
    plt.scatter(x=tt, y=tp, alpha=0.10, label="Train")

    m = max(vt.max(), tp.max(), vp.max(), vt.max())

    plt.plot([0, m], [0, m], label="y = x", alpha=0.5)
    plt.ylabel("Predictions")
    plt.xlabel("Ground Truth")

    # corrcords = (0.0, round(m*0.8))
    corrcords = (-1.0, -4)
    annotation_string = annotate_string([train_stats, test_stats])

    plt.annotate(
        annotation_string,
        corrcords,
        xycoords="data",
        fontsize=8,
        family="monospace",
        annotation_clip=False,
    )
    plt.title(f"{model_name}", fontsize=22)
    return fig, annotation_string


def get_summary(learner, model_name="", dls=None):
    if dls is None:
        tp, tt = learner.get_preds(
            ds_idx=0,
            act=None,
            with_input=False,
            with_loss=False,
            reorder=False,
            with_decoded=False,
        )
        tt = tt.squeeze().numpy()
        tp = tp.squeeze().numpy()

        vp, vt = learner.get_preds(
            ds_idxs=1,
            act=None,
            with_input=False,
            with_loss=False,
            reorder=False,
            with_decoded=False,
        )
        vt = vt.squeeze().numpy()
        vp = vp.squeeze().numpy()
    else:
        tp, tt = learner.get_preds(
            dl=dls.train,
            act=None,
            with_input=False,
            with_loss=False,
            reorder=False,
            with_decoded=False,
        )
        tt = tt.squeeze().numpy(),
        tp  = tp.squeeze().numpy()
        vp, vt = learner.get_preds(
            dl=dls.valid,
            act=None,
            with_input=False,
            with_loss=False,
            reorder=False,
            with_decoded=False,
        )
        vt = vt.squeeze().numpy()
        vp = vp.squeeze().numpy()

    return plot_regression(tp, tt, vp, vt, model_name=model_name)

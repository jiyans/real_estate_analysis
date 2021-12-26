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
    fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(5, 4))
    plt.scatter(x=targs, y=preds, alpha=0.10)
    plt.scatter(x=targs, y=preds, alpha=0.10)
    m = max(targs.max(), preds.max())
    corrcords = (0.0, round(m*0.8))
    
    plt.plot([0, m], [0, m], label="y = x", alpha=0.6)
    plt.legend(fontsize=15)
    plt.ylabel("Predictions", fontsize=18)
    plt.xlabel("Ground Truth", fontsize=18)
    annotation_string=annotate_string([trainstats])
    # plt.annotate(annotation_string,
    #              corrcords,
    #             fontsize=15, family="monospace")
    plt.title(f"{model_name}" , fontsize=22)
    return fig

def plot_regression(train_preds,train_targs, test_preds, test_targs,model_name=""):
    # Preds steals the indices, so we remove them from targs as
    train_targs = pd.Series(train_targs).reset_index(drop=True)
    train_preds = pd.Series(train_preds).reset_index(drop=True)

    test_targs = pd.Series(test_targs).reset_index(drop=True)
    test_preds = pd.Series(test_preds).reset_index(drop=True)

    mse_test = mean_squared_error(test_preds, test_targs)
    mse_train = mean_squared_error(train_preds, train_targs)
    N_test = test_preds.shape[0]
    N_train = train_preds.shape[0]
    corr_test = test_preds.corr(test_targs)
    corr_train = train_preds.corr(train_targs)

    test_stats =  Stats(corr=corr_test, mse=mse_test,rmse=np.sqrt(mse_test), N=N_test)
    train_stats=  Stats(corr=corr_train, mse=mse_train,rmse=np.sqrt(mse_train), N=N_train)

    plt.figure(figsize=(5, 4))
    plt.scatter(x=test_targs, y=test_preds, alpha=0.10, label="Test")
    plt.scatter(x=train_targs, y=train_preds, alpha=0.10, label="Train")

    m = max(test_targs.max(), train_preds.max(), test_preds.max(), test_targs.max())

    plt.plot([0, m], [0, m], label="y = x", alpha=0.5)
    plt.ylabel("Predictions")
    plt.xlabel("Ground Truth")

    corrcords = (0.0, round(m*0.8))
    annotation_string=annotate_string([test_stats, train_stats])

    # plt.annotate(annotation_string,
    #              corrcords,
    #             fontsize=15, family="monospace")
    return plt.title(f"{model_name}" , fontsize=22)

def get_summary(learner, model_name="", dls=None):
  if dls is None:
    tr_preds,tr_targs = learner.get_preds(ds_idx=0)
    te_preds,te_targs = learner.get_preds(ds_idx=1)
  else:
    tr_preds,tr_targs = learner.get_preds(dl=dls.train)
    te_preds,te_targs = learner.get_preds(dl=dls.valid)

  test_preds  = te_preds.squeeze().numpy()
  test_targs  = te_targs.squeeze().numpy()
  train_preds = tr_preds.squeeze().numpy()
  train_targs = tr_targs.squeeze().numpy()

  return plot_regression(train_preds,
                  train_targs,
                  test_preds,
                  test_targs,
                  model_name=model_name)

from sklearn.model_selection import LeaveOneOut
from sklearn import metrics

from Caspase_8_Analysis.metrics_utils import do_evaluations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function implementing Leave One Out cross-val. Due to only having 6 active compounds, I want to get as much info
# from them when training
# Returns: a 2 row list. First row contains the 1, 0 predictions of the ith sample,
# 2nd row the certainty (probability) of the ith prediction
def performLOO(classifier, X, Y, probsExist):
    print("\n>>> Performing LOO")

    loo = LeaveOneOut()
    predict_results = list()
    predict_probs = list()
    for train_indexes, test_index in loo.split(X):
        X_train = X.iloc[train_indexes, :]
        Y_train = Y.iloc[train_indexes]

        X_test = X.iloc[test_index, :]

        classifier.fit(np.array(X_train), np.array(Y_train))
        pred_y = classifier.predict(X_test)
        predict_results.append(pred_y[0])
        print(predict_results)
        if(probsExist):
            pred_y_prob = classifier.predict_proba(X_test)
            print(pred_y_prob)
            predict_probs.append([pred_y_prob[0][0], pred_y_prob[0][1]])

    print(np.array(predict_probs))
    do_evaluations(Y, predict_results, np.array(predict_probs), probs_exist=probsExist, displayed_name="LOO", show_plots=probsExist)

    return pred_y
from sklearn.model_selection import LeaveOneOut

from Caspase_8_Analysis.metrics_utils import doEvaluations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function implementing Leave One Out cross-val. Due to only having 6 active compounds, I want to get as much info
# from them when training
# Returns: a 2 row list. First row contains the 1, 0 predictions of the ith sample,
# 2nd row the certainty (probability) of the ith prediction
def performLOO(classifier, X, Y, probsExist):
    print()
    print(">>> Performing LOO")

    loo = LeaveOneOut()
    predict_results = list()
    predict_probs = list()
    for train_indexes, test_index in loo.split(X):
        X_train = X.iloc[train_indexes, :]
        Y_train = Y.iloc[train_indexes]

        X_test = X.iloc[test_index, :]

        classifier.fit(X_train, Y_train)
        pred_y = classifier.predict(X_test)
        predict_results.append(pred_y)

        if(probsExist):
            pred_y_prob = classifier.predict_proba(X_test)
            predict_probs.append([pred_y_prob[0][0], pred_y_prob[0][1]])



    doEvaluations(Y, predict_results, predict_probs, probs_exist=probsExist, displayed_name="LOO", show_plots=probsExist)

    return pred_y
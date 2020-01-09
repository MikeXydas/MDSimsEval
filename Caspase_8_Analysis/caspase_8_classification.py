from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Initialize indexes of features(1D, 2D, 3D, MD)
feature_groups = {}
feature_groups['1D_2D'] = list(np.arange(8))
feature_groups['3D'] = list(np.arange(8, 16))
feature_groups['MD'] = list(np.arange(16, 24))

# Reading the full dataset
df = pd.read_csv("../datasets/caspase_8.csv", sep=",")

# Inputs training_rows, indexes of wanted features
def split_dataset(df, training_rows, feature_indexes):
    print(feature_indexes)
    train_df_X = df.iloc[:training_rows, feature_indexes]
    train_df_Y = df.iloc[:training_rows, -1]

    # Everything not in train will be on test(validation) set
    test_df_X = df.iloc[training_rows:, feature_indexes]
    test_df_Y = df.iloc[training_rows:, -1]

    return train_df_X, train_df_Y, test_df_X, test_df_Y

# Choosing features and creating train and test sets
# Paper Selection: 29 training, 14 testing (4 actives in train, 2 actives in test)
features_selected = feature_groups['1D_2D'] + feature_groups['3D'] + feature_groups['MD']
train_df_X, train_df_Y, test_df_X, test_df_Y = split_dataset(df, 29, features_selected)

# Fitting on chosen method
logreg = LogisticRegression()
logreg.fit(train_df_X, train_df_Y)

# Predicting
pred_y = logreg.predict(test_df_X)

# Metrics
print(metrics.accuracy_score(test_df_Y, pred_y))
fpr, tpr, thresholds = metrics.roc_curve(test_df_Y, pred_y)
print(metrics.auc(fpr, tpr))


#print(train_df)
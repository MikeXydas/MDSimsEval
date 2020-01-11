from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

from sklearn import datasets

from Caspase_8_Analysis.metrics_utils import do_evaluations, plot_confusion, plot_scatter_points
from Caspase_8_Analysis.loo import performLOO

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

    train_df_X = df.iloc[:training_rows, feature_indexes]
    train_df_Y = df.iloc[:training_rows, -1]

    # Everything not in train will be on test(validation) set
    test_df_X = df.iloc[training_rows:, feature_indexes]
    test_df_Y = df.iloc[training_rows:, -1]

    #Get concatenated new X
    X_df = pd.concat([train_df_X, test_df_X])
    Y_df = pd.concat([train_df_Y, test_df_Y])
    #We need positive values to run SelectKBest
    normalized_df = (X_df - X_df.min()) / (X_df.max() - X_df.min())

    # Standarize data
    scaler = StandardScaler()
    standarized_df = scaler.fit_transform(np.array(X_df))   # Standarization

    # Calculate ANOVA f value and select top-k features
    # ANOVA_feature_selector = SelectKBest(f_classif, k=6).fit(standarized_df, Y_df)
    # final_features = ANOVA_feature_selector.get_support(indices=True)
    # features_df = pd.DataFrame(ANOVA_feature_selector.transform(standarized_df))

    # Perform Principal Component Analysis
    pca = PCA(n_components=2)
    final_features = np.zeros(pca.get_params()['n_components'])     # Needed to check condition if we can 2D plot
    features_df = pd.DataFrame(pca.fit_transform(standarized_df))  # Fit PCA
    features_df.columns = ["PC" + str(i) for i in range(1, len(features_df.columns) + 1)] # Give name to the columns
    comps = pd.DataFrame(pca.components_, columns=X_df.columns)     # Components (eigenvetors)
    variances = pd.DataFrame(pca.explained_variance_, columns=['Variance']) # Variances (eigenvalues)

    # Save components and variances in a csv
    comps = pd.concat([comps, variances], axis=1)
    comps.to_csv(path_or_buf='resources/PCA_components.csv', sep=',')

    plot_scatter_points(pd.concat([features_df, Y_df], axis=1), [0, 1], ["Inactive", "Active"], title="PCA Variance", show_ids=True)

    final_train_X = features_df.iloc[:training_rows, :]
    final_test_X = features_df.iloc[training_rows:, :]

    return final_train_X, train_df_Y, final_test_X, test_df_Y


# Choosing features and creating train and test sets
# Paper Selection: 29 training, 14 testing (4 actives in train, 2 actives in test)
features_selected = feature_groups['3D'] + feature_groups['MD']
train_df_X, train_df_Y, test_df_X, test_df_Y = split_dataset(df, 29, features_selected)

# Fitting on chosen method
#clf = RandomForestClassifier(class_weight={1:4}, n_estimators=7, criterion='entropy', max_features='sqrt', max_depth=60)
#clf = SVC(class_weight={1:6}, gamma='auto', tol=1e-5)

clf = LogisticRegression(class_weight={1:4}, solver='liblinear', penalty='l1', tol=5e-5, max_iter=10000)
#clf = KNeighborsClassifier(n_neighbors=1, algorithm='brute', p=2)


clf.fit(train_df_X, train_df_Y)

# Predicting
pred_y = clf.predict(test_df_X)
#pred_probs = clf.predict_proba(test_df_X)

do_evaluations(test_df_Y, pred_y, None, probs_exist=False, displayed_name="Paper Split", show_plots=True)
#plot_confusion(clf, "Caspase_8 Confusion", test_df_X, test_df_Y, ["Inactive", "Active"],
#               show_plot=False, print_confusion=True)

# Run Leave One Out cross-validation
pred_y = performLOO(clf, train_df_X.append(test_df_X), train_df_Y.append(test_df_Y), probsExist=False)


# # Number of trees in random forest
# n_estimators = list(np.arange(1,50, 5))
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [1, 3, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 3, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
#
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
#
# print(random_grid)
#
# clf = RandomForestClassifier()
#
# rf_random = RandomizedSearchCV(scoring='f1' ,estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# # Fit the random search model
# rf_random.fit(train_df_X.append(test_df_X), train_df_Y.append(test_df_Y))
#
# print(rf_random.best_params_)





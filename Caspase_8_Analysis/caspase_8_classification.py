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
feature_groups = {'2D': list(np.arange(8)), '3D': list(np.arange(8, 16)), 'MD': list(np.arange(16, 24))}

# Reading the full dataset
df = pd.read_csv("../datasets/caspase_8.csv", sep=",")

# Inputs training_rows, indexes of wanted features
# and perform feature selection
def split_dataset(df, training_rows, feature_indexes):

    train_df_X = df.iloc[:training_rows, feature_indexes]
    train_df_Y = df.iloc[:training_rows, -1]
    temp_cols = train_df_X.columns  # We need the column names for the components of PCA

    # Everything not in train will be on test(validation) set
    test_df_X = df.iloc[training_rows:, feature_indexes]
    test_df_Y = df.iloc[training_rows:, -1]

    # Normalize (We need positive values to run SelectKBest)
    # train_df_X = (train_df_X - train_df_X.min()) / (train_df_X.max() - train_df_X.min())
    # test_df_X = (test_df_X - test_df_X.min()) / (test_df_X.max() - test_df_X.min())

    # Standardize data
    scaler = StandardScaler()
    train_df_X = pd.DataFrame(scaler.fit_transform(np.array(train_df_X)), columns=train_df_X.columns)   # Standardization
    test_df_X = pd.DataFrame(scaler.transform(np.array(test_df_X)), columns=test_df_X.columns)

    features_df = train_df_X

    # Calculate ANOVA f value and select top-k features
    ANOVA_feature_selector = SelectKBest(f_classif, k=5).fit(features_df, train_df_Y)
    final_features_indexes = ANOVA_feature_selector.get_support(indices=True)
    features_df = pd.DataFrame(ANOVA_feature_selector.transform(features_df))
    pd.DataFrame([ANOVA_feature_selector.scores_], columns=temp_cols)\
        .round(4).to_csv(path_or_buf="resources/ANOVA_MD_scores_NULL.csv")       # Save scores of features
    final_features_names = [temp_cols[i] for i in final_features_indexes]
    features_df.columns = final_features_names
    # pd.DataFrame(final_features_names).to_csv(path_or_buf="resources/ANOVA_best_k_features.csv")
    test_df_X = test_df_X.iloc[:, final_features_indexes]   # Get only the selected features on test set
    temp_cols = features_df.columns

    # Perform Principal Component Analysis
    pca = PCA(n_components=3)
    final_features = np.zeros(pca.get_params()['n_components'])     # Needed to check condition if we can 2D plot
    features_df = pd.DataFrame(pca.fit_transform(features_df))  # Fit PCA
    features_df.columns = ["PC" + str(i) for i in range(1, pca.n_components_ + 1)] # Give name to the columns
    comps = pd.DataFrame(pca.components_, columns=temp_cols).round(4)     # Components (eigenvetors)
    variances = pd.DataFrame(pca.explained_variance_, columns=['Variance']).round(4) # Variances (eigenvalues)
    test_df_X = pca.transform(test_df_X)    # Apply the transformation on the test set
    # Save components and variances in a csv
    comps = pd.concat([comps, variances], axis=1)
    # comps.to_csv(path_or_buf='resources/PCA_components_MD_3D_no_20_23_NULL.csv', sep=',')

    # This will result in a plot if remaining features of features_df = 2
    plot_scatter_points(pd.concat([features_df, train_df_Y], axis=1), [0, 1],
                        ["Inactive", "Active"], title="PCA_F_Values_MD_3D", show_ids=True)

    # Return the features selected for training and evaluation
    final_train_X = features_df.iloc[:training_rows, :]
    final_test_X = features_df.iloc[training_rows:, :]

    return final_train_X, train_df_Y, test_df_X, test_df_Y


# Choosing features and creating train and test sets
# Paper Selection: 29 training, 14 testing (4 actives in train, 2 actives in test)
features_selected = feature_groups['MD'] + feature_groups['3D']
train_df_X, train_df_Y, test_df_X, test_df_Y = split_dataset(df, 29, features_selected)

# Fitting on chosen method
#clf = RandomForestClassifier(class_weight={1:5}, n_estimators=1, criterion='entropy',
#                                max_features='sqrt', max_depth=60)
# clf = RandomForestClassifier()
# clf = SVC(class_weight={1:6}, gamma='auto', tol=1e-5)
# clf = SVC()
# clf = LogisticRegression(class_weight={1:5}, solver='liblinear', penalty='l1', tol=5e-5, max_iter=10000)
# clf = LogisticRegression()
# clf = KNeighborsClassifier(n_neighbors=1, algorithm='brute', p=2)
clf = KNeighborsClassifier()

clf.fit(train_df_X, train_df_Y)

# Predicting on test set
print("\n>>> Evaluating on paper test split")
pred_y = clf.predict(test_df_X)
#pred_probs = clf.predict_proba(test_df_X)

do_evaluations(test_df_Y, pred_y, None, probs_exist=False, displayed_name="Paper Split", show_plots=True)
#plot_confusion(clf, "Caspase_8 Confusion", test_df_X, test_df_Y, ["Inactive", "Active"],
#               show_plot=False, print_confusion=True)

# Run Leave One Out cross-validation on training set
pred_y = performLOO(clf, train_df_X, train_df_Y, probsExist=False)



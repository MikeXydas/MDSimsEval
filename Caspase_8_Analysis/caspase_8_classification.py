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
from sklearn.model_selection import cross_validate

from sklearn import datasets

from Caspase_8_Analysis.metrics_utils import do_evaluations, plot_confusion, plot_scatter_points,performKFold
from Caspase_8_Analysis.loo import performLOO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set basic parameters for the experiment
exp_params = {
    "dataset_path": "../datasets/caspase_8csv",
    "pca_componentes": 8,   # if -1, then PCA is not applied
    "pca_loadings_path": "resources/PCA_components_MD_no_20_23.csv",
    "KBest_features": -1,    # if -1, then KBest is not applied
    "KBest_scores_path": "resources/ANOVA_MD_scores_BBB.csv",
    "scatter_plot_title": "PCA_MD_no_20_23",    # there will be a plot if features are 2D
    "feature_groups": ["MD"]              # Possible values: "2D", "3D", "MD"
}

# Initialize indexes of features(1D, 2D, 3D, MD)
feature_groups = {'2D': list(np.arange(8)), '3D': list(np.arange(8, 16)), 'MD': list(np.arange(16, 24))}

# Reading the full dataset
df = pd.read_csv(exp_params['dataset_path'], sep=",")
# df = df.sample(frac=1)

# Inputs training_rows, indexes of wanted features
# and perform feature selection
def split_dataset(df, training_rows, feature_indexes):

    train_df_X = df.iloc[:training_rows, feature_indexes]
    train_df_Y = df.iloc[:training_rows, -1]
    temp_cols = train_df_X.columns  # We need the column names for the components of PCA

    # Everything not in training will be on test set
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
    if(exp_params['KBest_features'] != -1):
        ANOVA_feature_selector = SelectKBest(f_classif, k=exp_params['KBest_features']).fit(features_df, train_df_Y)
        final_features_indexes = ANOVA_feature_selector.get_support(indices=True)       # Indices of the top-k features
        features_df = pd.DataFrame(ANOVA_feature_selector.transform(features_df))       # A df containing the surviving features of shape [topK, training_rows]
        pd.DataFrame([ANOVA_feature_selector.scores_], columns=temp_cols)\
            .round(4).to_csv(path_or_buf=exp_params['KBest_scores_path'])               # Save scores of features
        features_df.columns = [temp_cols[i] for i in final_features_indexes]            # Get which column names survived
        test_df_X = test_df_X.iloc[:, final_features_indexes]                           # Get only the selected features on test set
        temp_cols = features_df.columns

    # Perform Principal Component Analysis
    if(exp_params['pca_componentes'] != -1):
        pca = PCA(n_components=exp_params['pca_componentes'])
        final_features = np.zeros(pca.get_params()['n_components'])     # Needed to check condition if we can 2D plot
        features_df = pd.DataFrame(pca.fit_transform(features_df))      # Fit and transform on the training set
        features_df.columns = ["PC" + str(i) for i in range(1, pca.n_components_ + 1)]          # Give name to the columns (PC1, PC2, ...)
        comps = pd.DataFrame(pca.components_, columns=temp_cols).round(4)                       # Loading Values
        variances = pd.DataFrame(pca.explained_variance_, columns=['Variance']).round(4)        # Variances
        test_df_X = pca.transform(test_df_X)                            # Apply ONLY the transformation on the test set
        # Save loading values and variances in a csv
        comps = pd.concat([comps, variances], axis=1)
        comps = pd.concat([pd.DataFrame(["PC" + str(i) for i in range(1, exp_params['pca_componentes'] + 1)], columns=["PCs"]), comps], axis=1)
        comps.to_csv(path_or_buf=exp_params['pca_loadings_path'], sep=',')

    # This will result in a plot if remaining features of features_df = 2
    plot_scatter_points(pd.concat([features_df, train_df_Y], axis=1), [0, 1],
                        ["Inactive", "Active"], title=exp_params['scatter_plot_title'], show_ids=True)

    final_train_X = features_df.iloc[:training_rows, :]

    # Return the selected features for training and testing
    return final_train_X, train_df_Y, test_df_X, test_df_Y


# Choosing features and creating train and test sets
# Paper Selection: 29 training, 14 testing (4 actives in train, 2 actives in test)
features_selected = np.array([feature_groups[i] for i in exp_params['feature_groups']]).flatten()
train_df_X, train_df_Y, test_df_X, test_df_Y = split_dataset(df, 29, features_selected)

# Fitting on chosen method
# clf = RandomForestClassifier(class_weight={0:0.1, 1:0.9}, n_estimators=100, criterion='entropy',
#                                max_features='log2')
clf = RandomForestClassifier(random_state=1, class_weight={1:9}, max_features='log2')
# clf = SVC(class_weight={1:6}, gamma='auto', tol=1e-5)
# clf = SVC()
# clf = LogisticRegression(class_weight="balanced", solver='liblinear', penalty='l1', tol=5e-5, max_iter=10000)
# clf = LogisticRegression()
# clf = KNeighborsClassifier(n_neighbors=1, algorithm='brute', p=2)
# clf = KNeighborsClassifier()

clf.fit(train_df_X, train_df_Y)

# Predicting on test set
print("\n>>> Evaluating on paper test split")
pred_y = clf.predict(test_df_X)
pred_y_probs = clf.predict_proba(test_df_X)
do_evaluations(test_df_Y, pred_y, pred_y_probs, probs_exist=True, displayed_name="Paper Split", show_plots=True)

# Doing k-fold (on paper 10-fold was used)
# performKFold(clf, np.array(train_df_X), np.array(train_df_Y), 10)


#plot_confusion(clf, "Caspase_8 Confusion", test_df_X, test_df_Y, ["Inactive", "Active"],
#               show_plot=False, print_confusion=True)



# Run Leave One Out cross-validation on training set
pred_y = performLOO(clf, train_df_X, train_df_Y, probsExist=True)


# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 1, stop = 100, num = 2)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(5, 110, num = 5)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [1, 2, 3, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 3, 4, 5]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# class_weight=[{1:5}, {1:4}, {1:3}, {1:2}, {1:1}, {1:6}, {1:7}]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap,
#                'class_weight': class_weight
#                }
#
# rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 1000, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring='f1')
# rf_random.fit(train_df_X, train_df_Y)
# print(rf_random.best_params_)

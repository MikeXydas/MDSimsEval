from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix

from sklearn import datasets

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

    return train_df_X, train_df_Y, test_df_X, test_df_Y


# Helper function that prints the confusion matrix
def plot_confusion(classifier, title, test_X, test_Y, class_labels, show_plot, print_confusion):
    disp = plot_confusion_matrix(classifier, test_X, test_Y,
                                 display_labels=class_labels,
                                 cmap=plt.cm.Blues, normalize=None)
    disp.ax_.set_title(title)
    if (show_plot):
        plt.show()
    if (print_confusion):
        print(title)
        print(disp.confusion_matrix)


# Print the difference of probabilities
def plot_probabilities(classifier, test_X, test_Y, class_labels, show_plot, print_probabilities):
    probs = np.array(classifier.predict_proba(test_X)).transpose()
    y_labels = ["Active" if i != 0 else "Inactive" for i in test_Y]

    if(print_probabilities):
        print(probs.transpose())

    if(show_plot):
        plt.plot()
        plt.plot(np.arange(len(probs[0])), probs[0], label=class_labels[0], marker='o')
        plt.plot(np.arange(len(probs[0])), probs[1], label=class_labels[1], marker='o')
        plt.xticks(np.arange(len(probs[0])), y_labels, rotation='60')
        plt.xlabel('Label')
        plt.ylabel('Probability')
        plt.title('Difference of Probability (each datapoint)')
        plt.legend()
        plt.tight_layout()
        plt.show()

# Choosing features and creating train and test sets
# Paper Selection: 29 training, 14 testing (4 actives in train, 2 actives in test)
features_selected = feature_groups['MD']
#features_selected=[0, 1, 2, 3]
train_df_X, train_df_Y, test_df_X, test_df_Y = split_dataset(df, 29, features_selected)

# Fitting on chosen method
clf = RandomForestClassifier(random_state=0)
clf.fit(train_df_X, train_df_Y)

# Predicting
pred_y = clf.predict(test_df_X)

# Evaluating Results
print(">>> Accuracy: " + str(metrics.accuracy_score(test_df_Y, pred_y)))

print(">>> Recall: " + str(metrics.recall_score(test_df_Y, pred_y)))

fpr, tpr, thresholds = metrics.roc_curve(test_df_Y, pred_y)
print(">>> AUC: " + str(metrics.auc(fpr, tpr)))

plot_probabilities(clf, test_df_X, test_df_Y, ["Inactive", "Active"], show_plot=True, print_probabilities=False)
plot_confusion(clf, "Caspase_8 Confusion", test_df_X, test_df_Y, ["Inactive", "Active"],
               show_plot=True, print_confusion=True)

